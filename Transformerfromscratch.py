import math
import random

# ============= Math Utilities =============
def dot_product(v1, v2):
    """Compute dot product of two vectors"""
    return sum(a * b for a, b in zip(v1, v2))

def matrix_multiply(A, B):
    """Multiply two matrices"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    if cols_A != rows_B:
        raise ValueError(f"Cannot multiply matrices: {cols_A} != {rows_B}")
    
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

def transpose(matrix):
    """Transpose a matrix"""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def softmax(x):
    """Compute softmax of a vector"""
    exp_x = [math.exp(i - max(x)) for i in x]
    sum_exp = sum(exp_x)
    return [i / sum_exp for i in exp_x]

def layer_norm(x, eps=1e-6):
    """Layer normalization"""
    mean = sum(x) / len(x)
    variance = sum((i - mean) ** 2 for i in x) / len(x)
    return [(i - mean) / math.sqrt(variance + eps) for i in x]

def gelu(x):
    """GELU activation function"""
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

# ============= Weight Initialization =============
def xavier_init(rows, cols):
    """Xavier initialization for weights"""
    limit = math.sqrt(6 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]

def zeros(rows, cols):
    """Create a matrix of zeros"""
    return [[0 for _ in range(cols)] for _ in range(rows)]

# ============= Attention Mechanism =============
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices
        self.W_q = xavier_init(d_model, d_model)
        self.W_k = xavier_init(d_model, d_model)
        self.W_v = xavier_init(d_model, d_model)
        self.W_o = xavier_init(d_model, d_model)
    
    def split_heads(self, x):
        """Split the last dimension into (num_heads, d_k)"""
        seq_len = len(x)
        result = []
        for i in range(seq_len):
            heads = []
            for h in range(self.num_heads):
                head = x[i][h * self.d_k:(h + 1) * self.d_k]
                heads.append(head)
            result.append(heads)
        return result
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention"""
        seq_len = len(Q)
        d_k = len(Q[0])
        
        # Compute attention scores
        scores = [[dot_product(Q[i], K[j]) / math.sqrt(d_k) 
                   for j in range(seq_len)] for i in range(seq_len)]
        
        # Apply mask if provided
        if mask:
            for i in range(seq_len):
                for j in range(seq_len):
                    if mask[i][j] == 0:
                        scores[i][j] = -1e9
        
        # Apply softmax
        attention_weights = [softmax(row) for row in scores]
        
        # Compute output
        output = []
        for i in range(seq_len):
            out_vec = [0] * d_k
            for j in range(seq_len):
                for k in range(d_k):
                    out_vec[k] += attention_weights[i][j] * V[j][k]
            output.append(out_vec)
        
        return output, attention_weights
    
    def forward(self, x, mask=None):
        """Forward pass of multi-head attention"""
        seq_len = len(x)
        
        # Linear projections
        Q = matrix_multiply(x, self.W_q)
        K = matrix_multiply(x, self.W_k)
        V = matrix_multiply(x, self.W_v)
        
        # Split into multiple heads
        Q_heads = self.split_heads(Q)
        K_heads = self.split_heads(K)
        V_heads = self.split_heads(V)
        
        # Apply attention for each head
        head_outputs = []
        for h in range(self.num_heads):
            Q_h = [Q_heads[i][h] for i in range(seq_len)]
            K_h = [K_heads[i][h] for i in range(seq_len)]
            V_h = [V_heads[i][h] for i in range(seq_len)]
            
            head_out, _ = self.scaled_dot_product_attention(Q_h, K_h, V_h, mask)
            head_outputs.append(head_out)
        
        # Concatenate heads
        concat = []
        for i in range(seq_len):
            concat_vec = []
            for h in range(self.num_heads):
                concat_vec.extend(head_outputs[h][i])
            concat.append(concat_vec)
        
        # Final linear projection
        output = matrix_multiply(concat, self.W_o)
        return output

# ============= Feed-Forward Network =============
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = xavier_init(d_model, d_ff)
        self.b1 = [0] * d_ff
        self.W2 = xavier_init(d_ff, d_model)
        self.b2 = [0] * d_model
    
    def forward(self, x):
        """Forward pass: FFN(x) = GELU(xW1 + b1)W2 + b2"""
        seq_len = len(x)
        
        # First layer
        hidden = matrix_multiply(x, self.W1)
        for i in range(seq_len):
            for j in range(len(self.b1)):
                hidden[i][j] += self.b1[j]
        
        # Apply GELU activation
        for i in range(seq_len):
            for j in range(len(hidden[i])):
                hidden[i][j] = gelu(hidden[i][j])
        
        # Second layer
        output = matrix_multiply(hidden, self.W2)
        for i in range(seq_len):
            for j in range(len(self.b2)):
                output[i][j] += self.b2[j]
        
        return output

# ============= Transformer Block =============
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.gamma1 = [1.0] * d_model
        self.beta1 = [0.0] * d_model
        self.gamma2 = [1.0] * d_model
        self.beta2 = [0.0] * d_model
    
    def forward(self, x, mask=None):
        """Forward pass with residual connections and layer norm"""
        seq_len = len(x)
        d_model = len(x[0])
        
        # Self-attention with residual connection
        attn_out = self.attention.forward(x, mask)
        for i in range(seq_len):
            for j in range(d_model):
                attn_out[i][j] += x[i][j]
        
        # Layer norm 1
        for i in range(seq_len):
            attn_out[i] = layer_norm(attn_out[i])
            for j in range(d_model):
                attn_out[i][j] = attn_out[i][j] * self.gamma1[j] + self.beta1[j]
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward.forward(attn_out)
        for i in range(seq_len):
            for j in range(d_model):
                ff_out[i][j] += attn_out[i][j]
        
        # Layer norm 2
        for i in range(seq_len):
            ff_out[i] = layer_norm(ff_out[i])
            for j in range(d_model):
                ff_out[i][j] = ff_out[i][j] * self.gamma2[j] + self.beta2[j]
        
        return ff_out

# ============= Complete Transformer Model =============
class Transformer:
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_len=512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = xavier_init(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        
        # Output layer
        self.output_layer = xavier_init(d_model, vocab_size)
    
    def create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encodings"""
        pe = [[0 for _ in range(d_model)] for _ in range(max_len)]
        
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos][i] = math.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pe[pos][i + 1] = math.cos(pos / (10000 ** (i / d_model)))
        
        return pe
    
    def embed(self, token_ids):
        """Embed tokens and add positional encoding"""
        seq_len = len(token_ids)
        embeddings = []
        
        for i, token_id in enumerate(token_ids):
            # Get token embedding
            token_emb = self.token_embedding[token_id][:]
            
            # Add positional encoding
            for j in range(self.d_model):
                token_emb[j] += self.positional_encoding[i][j]
            
            embeddings.append(token_emb)
        
        return embeddings
    
    def forward(self, token_ids, mask=None):
        """Forward pass through the transformer"""
        # Embed tokens
        x = self.embed(token_ids)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)
        
        # Project to vocabulary
        logits = matrix_multiply(x, self.output_layer)
        
        return logits

# ============= Example Usage =============
if __name__ == "__main__":
    # Create a small transformer
    vocab_size = 1000
    model = Transformer(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_seq_len=50
    )
    
    # Example input: sequence of token IDs
    input_tokens = [5, 10, 15, 20, 25]
    
    print("Transformer Model (Pure Python)")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Model dimension: {model.d_model}")
    print(f"Number of layers: {len(model.blocks)}")
    print(f"\nInput tokens: {input_tokens}")
    
    # Forward pass
    output = model.forward(input_tokens)
    
    print(f"Output shape: {len(output)} x {len(output[0])}")
    print(f"\nFirst position logits (first 10): {output[0][:10]}")
    
    # Get predictions
    predictions = [softmax(logits) for logits in output]
    predicted_tokens = [max(range(len(probs)), key=lambda i: probs[i]) for probs in predictions]
    
    print(f"\nPredicted tokens: {predicted_tokens}")