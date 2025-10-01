# transformer-ANN-model-from-scratch
With pure python because it means pure logic of transformer and its main functions like multi head self attention , we may think it as pseudeo code

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

it is not trained because of that predictions may be unexpected (author(me) has talked about py file from scratch)


Token IDs: [5, 10, 15, 20, 25]
    ↓
┌───────────────────────────┐
│   TOKEN EMBEDDING         │  Convert IDs to vectors
│   + POSITIONAL ENCODING   │  Add position info
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│  TRANSFORMER BLOCK 1      │
│  ┌─────────────────────┐  │
│  │ Multi-Head Attention│  │  Learn relationships
│  │  + Residual + Norm  │  │
│  └─────────────────────┘  │
│  ┌─────────────────────┐  │
│  │  Feed-Forward       │  │  Transform features
│  │  + Residual + Norm  │  │
│  └─────────────────────┘  │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│  TRANSFORMER BLOCK 2      │  (Same structure)
│  (Attention + FFN)        │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│   OUTPUT PROJECTION       │  Project to vocabulary
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│       SOFTMAX             │  Convert to probabilities
└───────────────────────────┘
    ↓
Predicted Tokens: [992, 992, 992, 992, 992]
