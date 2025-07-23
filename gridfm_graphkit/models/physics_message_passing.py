import torch
from torch import nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from gridfm_graphkit.datasets.globals import PD, QD, PG, QG, VM, VA, PQ, PV, REF, FEATURES_IDX, BUS_TYPES

class MessagePassingLayer(nn.Module):
    """
    Message passing layer for power grid physics-based neural solver.
    """
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        # Standard message generation networks
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        
        # P and V specific message networks
        self.p_edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
        )
        
        self.v_edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
        )
        
        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Performs one round of message passing with special handling for P and V values.
        """
        src, dst = edge_index
        # Gather source and destination node features
        src_features = x[src]
        dst_features = x[dst]
        
        # Create message inputs by combining source, destination and edge features
        message_inputs = torch.cat([src_features, dst_features, edge_attr], dim=-1)
        
        # Generate standard messages
        messages = self.edge_mlp(message_inputs)
        
        # Generate P and V specific messages
        p_messages = self.p_edge_mlp(message_inputs)
        v_messages = self.v_edge_mlp(message_inputs)
        
        # Combine messages (allocate specific portions of the hidden dimension for P and V)
        hidden_dim = messages.size(-1)
        combined_messages = torch.cat([
            messages[:, :hidden_dim - p_messages.size(-1) - v_messages.size(-1)],
            p_messages,
            v_messages
        ], dim=1)
        
        # Aggregate messages using PyG's built-in mechanisms
        row, col = edge_index
        aggr_messages = pyg_nn.aggr.SumAggregation()(combined_messages, col, size=x.size(0))
        
        # Combine node features with aggregated messages
        combined_features = torch.cat([x, aggr_messages], dim=-1)
        
        # Update node representations
        updated_x = self.node_mlp(combined_features)
        
        return updated_x


class PhysicsMessagePassingModel(nn.Module):
    """
    Graph Neural Solver for power systems using physics-based message passing.
    Extends the GraphNeuralSolver approach to predict all power grid quantities.
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim,
                 output_dim, 
                 edge_dim,
                 num_layers=5,
                 mask_dim=6,
                 mask_value=None,
                 learn_mask=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.mask_dim = mask_dim
        
        # Initial embedding layer
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        
        # Output decoder
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Physics-based correction layer - computes equilibrium violations
        self.equilibrium_correction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim//2, output_dim),
        )
        
        # Learnable mask value if needed
        self.learn_mask = learn_mask
        if mask_value is not None:
            self.mask_value = mask_value
        else:
            self.mask_value = torch.zeros(mask_dim)
            if learn_mask:
                self.mask_value = nn.Parameter(self.mask_value)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def calculate_power_flow_equations(self, pred, edge_index, edge_attr):
        """
        Calculate power flow equations for physics-based corrections.
        Implements similar functionality to EquilibriumViolation.error_tensor in GraphNeuralSolver.
        """
        # Extract voltage magnitudes and angles
        vm = pred[:, FEATURES_IDX[VM]]  # Voltage magnitude
        va = pred[:, FEATURES_IDX[VA]]  # Voltage angle
        pg = pred[:, FEATURES_IDX[PG]]  # Active power generation
        qg = pred[:, FEATURES_IDX[QG]]  # Reactive power generation
        pd = pred[:, FEATURES_IDX[PD]]  # Active power demand
        qd = pred[:, FEATURES_IDX[QD]]  # Reactive power demand
        
        # Extract grid parameters from edge attributes (conductance and susceptance)
        g_real = edge_attr[:, 0]  # Conductance
        b_imag = edge_attr[:, 1]  # Susceptance
        
        # Get source and destination nodes for each edge
        src, dst = edge_index
        
        # Calculate active and reactive power injections for each bus
        p_calc = torch.zeros_like(vm)
        q_calc = torch.zeros_like(vm)
        
        # For each edge, calculate power flow contribution using AC power flow equations
        for i in range(edge_index.shape[1]):
            from_idx = src[i]
            to_idx = dst[i]
            g = g_real[i]
            b = b_imag[i]
            
            # Voltage magnitudes at both ends
            v_from = vm[from_idx]
            v_to = vm[to_idx]
            
            # Voltage angle difference
            angle_diff = va[from_idx] - va[to_idx]
            
            # Calculate active and reactive power flow contributions (AC power flow equations)
            p_flow = v_from * v_to * (g * torch.cos(angle_diff) + b * torch.sin(angle_diff))
            q_flow = v_from * v_to * (g * torch.sin(angle_diff) - b * torch.cos(angle_diff))
            
            # Accumulate power injections at the from_bus
            p_calc[from_idx] += p_flow
            q_calc[from_idx] += q_flow
        
        # Calculate power imbalances (should be zero in a valid power flow solution)
        p_imbalance = pg - pd - p_calc
        q_imbalance = qg - qd - q_calc
        
        # Return imbalances for both active and reactive power
        return torch.stack([p_imbalance, q_imbalance], dim=1)
    
    def forward(self, x, pe, edge_index, edge_attr, batch=None):
        """
        Forward pass through the physics-based message passing model.
        """
        # Extract bus types from input features
        mask_PQ = x[:, PQ] == 1  # PQ buses
        mask_PV = x[:, PV] == 1  # PV buses
        mask_REF = x[:, REF] == 1  # Reference buses
        
        # Initial node embeddings
        h = self.node_encoder(x)
        
        # Store initial prediction for correction
        initial_h = h.clone()
        
        # Iterative message passing
        for i in range(self.num_layers):
            h = self.mp_layers[i](h, edge_index, edge_attr)
        
        # Get initial prediction
        pred = self.node_decoder(h)
        
        # Apply physics-based correction iterations
        for _ in range(3):  # Number of correction steps
            # Compute equilibrium violations/residuals
            physics_correction = self.equilibrium_correction(h)
            
            # Apply bus-type specific scaling to corrections
            # Different types of buses have different variables that can be adjusted
            scaled_correction = physics_correction.clone()
            
            # PQ buses: VM and VA are typically predicted
            # PV buses: VA and QG are typically predicted (VM is fixed)
            # REF buses: PG and QG are typically predicted (VM and VA are fixed)
            
            # Don't correct VM for PV and REF buses
            scaled_correction[mask_PV, FEATURES_IDX[VM]] = 0
            scaled_correction[mask_REF, FEATURES_IDX[VM]] = 0
            
            # Don't correct VA for REF buses
            scaled_correction[mask_REF, FEATURES_IDX[VA]] = 0
            
            # Apply physics-based corrections
            pred = pred - scaled_correction
            
            # Optionally incorporate power flow equation corrections
            if _ == 2:  # On the last iteration
                # Calculate power flow imbalances
                imbalances = self.calculate_power_flow_equations(pred, edge_index, edge_attr)
                
                # Apply targeted corrections based on bus type
                # For PQ buses - adjust VM based on reactive power imbalance
                correction_factor = 0.05
                pred[mask_PQ, FEATURES_IDX[VM]] += imbalances[mask_PQ, 1] * correction_factor
                
                # For PV buses - adjust QG based on reactive power imbalance
                pred[mask_PV, FEATURES_IDX[QG]] += imbalances[mask_PV, 1] * correction_factor
                
                # For all buses - apply small correction to active power generation
                pred[:, FEATURES_IDX[PG]] += imbalances[:, 0] * correction_factor
        
        return pred


class PBEMessagePassingModel(PhysicsMessagePassingModel):
    """
    Extension of PhysicsMessagePassingModel with explicit Power Balance Equation constraints.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add specialized layers for PBE constraint enforcement
        self.pbe_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 2)  # For active and reactive power balance
        )
    
    def power_balance_correction(self, pred, edge_index, edge_attr, mask=None):
        """
        Apply power balance equation constraints explicitly to refine predictions
        """
        # Extract relevant quantities from predictions
        vm = pred[:, FEATURES_IDX[VM]]
        va = pred[:, FEATURES_IDX[VA]]
        pg = pred[:, FEATURES_IDX[PG]]
        qg = pred[:, FEATURES_IDX[QG]]
        pd = pred[:, FEATURES_IDX[PD]]
        qd = pred[:, FEATURES_IDX[QD]]
        
        # Extract grid parameters from edge attributes
        g_real = edge_attr[:, 0]  # Conductance
        b_imag = edge_attr[:, 1]  # Susceptance
        
        # Start with current predictions
        corrected_pred = pred.clone()
        
        # Get source and destination nodes for each edge
        src, dst = edge_index
        
        # Calculate power injections using AC power flow equations
        p_calc = torch.zeros_like(vm)
        q_calc = torch.zeros_like(vm)
        
        # For each edge, calculate power flow contribution
        for i in range(edge_index.shape[1]):
            from_idx = src[i]
            to_idx = dst[i]
            g = g_real[i]
            b = b_imag[i]
            
            # Voltage magnitudes at both ends
            v_from = vm[from_idx]
            v_to = vm[to_idx]
            
            # Voltage angle difference
            angle_diff = va[from_idx] - va[to_idx]
            
            # Calculate active and reactive power flow contributions
            p_flow = v_from * v_to * (g * torch.cos(angle_diff) + b * torch.sin(angle_diff))
            q_flow = v_from * v_to * (g * torch.sin(angle_diff) - b * torch.cos(angle_diff))
            
            # Accumulate power injections
            p_calc[from_idx] += p_flow
            q_calc[from_idx] += q_flow
        
        # If mask is provided, only correct masked values
        if mask is not None:
            # Find which P and V values were masked
            p_mask = torch.logical_or(
                mask[:, FEATURES_IDX[PG]], 
                mask[:, FEATURES_IDX[PD]]
            )
            v_mask = mask[:, FEATURES_IDX[VM]]
            
            # Calculate power imbalances
            p_imbalance = pg - pd - p_calc
            q_imbalance = qg - qd - q_calc
            
            # Apply corrections only to masked values
            # Active power correction
            p_correction_factor = 0.1  # Damping factor
            corrected_pred[p_mask, FEATURES_IDX[PG]] += p_imbalance[p_mask] * p_correction_factor
            
            # Voltage magnitude correction based on reactive power imbalance
            v_correction_factor = 0.05  # Damping factor
            corrected_pred[v_mask, FEATURES_IDX[VM]] += q_imbalance[v_mask] * v_correction_factor
        
        return corrected_pred
    
    def forward(self, x, pe, edge_index, edge_attr, batch=None):
        """
        Forward pass with explicit power balance constraint enforcement
        """
        # Get initial predictions from parent class
        pred = super().forward(x, pe, edge_index, edge_attr, batch)
        
        # Extract mask from input if available
        original_mask = None
        if hasattr(batch, 'mask'):
            original_mask = batch.mask
        
        # Apply power balance corrections
        pred = self.power_balance_correction(pred, edge_index, edge_attr, original_mask)
        
        # Extract bus types
        mask_PQ = x[:, PQ] == 1
        mask_PV = x[:, PV] == 1
        mask_REF = x[:, REF] == 1
        
        # Calculate power mismatches
        imbalances = self.calculate_power_flow_equations(pred, edge_index, edge_attr)
        
        # Apply final PBE-specific corrections based on bus type
        # These are additional fine-tuning corrections specific to power systems
        
        # For PQ buses: adjust VM based on reactive power mismatch
        pred[mask_PQ, FEATURES_IDX[VM]] += imbalances[mask_PQ, 1] * 0.02
        
        # For PV buses: adjust QG based on reactive power mismatch
        pred[mask_PV, FEATURES_IDX[QG]] += imbalances[mask_PV, 1] * 0.05
        
        # For REF buses: adjust PG based on active power mismatch
        pred[mask_REF, FEATURES_IDX[PG]] += imbalances[mask_REF, 0] * 0.05
        
        return pred