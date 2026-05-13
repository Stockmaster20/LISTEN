# SPCPNet: Understanding the Cocktail Party Effect in Wireless Signal Recognition

---
## This is official code of paper "SPCPNet: Understanding the Cocktail Party Effect in Wireless Signal Recognition". 

The motivation of our work: We draw inspiration from the cocktail party effect (top). In this cognitive process, the human brain first decouples mixed acoustic waves into distinct streams and then selectively recognizes the target information. Motivated by this mechanism, we design SPCPNet (bottom) to mimic this biological paradigm for wireless signal recognition.

![The motivation of our work: We draw inspiration from the cocktail party effect (top). In this cognitive process, the human brain first decouples mixed acoustic waves into distinct streams and then selectively recognizes the target information. Motivated by this mechanism, we design SPCPNet (bottom) to mimic this biological paradigm for wireless signal recognition.](motivation.png)

Illustration of the proposed SPCPNet.

![Illustration of the proposed SPCPNet.](method.png)

If you want to use SPCPNet, you can follow:
```python
    dummy_input = torch.randn(batch_size, 2, seq_length)
    model = SPCPNet(in_channels=2, num_stages=3, num_classes=num_classes, feature_dim=32)
    logits, S_k, L_k, X_feat = model(dummy_input)
    ...
