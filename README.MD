# miniSAM

This is an implementation of Meta's Segment Anything Model (SAM) built from scratch with Python and PyTorch tensors only (for now, no autograd but this might change). It consists of an image encoder using Masked Autoencoder (MAE) principles, a prompt encoder, and a lightweight mask decoder.

## Roadmap

### MAE (Masked Autoencoder) Image Encoder

Vision Transformer-based encoder that processes images into embeddings.

- [ ] Patch embedding layer (16x16 patches)
- [ ] Positional encoding implementation
- [ ] Multi-head self-attention blocks
- [ ] Forward pass
- [ ] Backward pass
- [ ] Testing (manual gradients vs. autograd) (not decided yet on this)
- [ ] Pre-training

### Prompt Encoder

Encodes a sparse point prompt into embedding space that the decoder can use.

- [ ] Point prompt encoding (with positive/negative labels)
- [ ] Positional encoding for sparse prompt
- [ ] Forward pass
- [ ] Testing against reference implementation

### Mask Decoder

Lightweight decoder that combines image embeddings with prompt embeddings to predict segmentation masks.

- [ ] Two-way transformer blocks (self-attention + cross-attention)
- [ ] IoU prediction head
- [ ] Mask prediction head (upsampling to original resolution)
- [ ] Forward pass
- [ ] Backward pass
- [ ] Testing (gradient verification, mask quality metrics)
- [ ] Multi-mask output support (3 masks with confidence scores)

## Resources

- [An Image is Worth 16x16 Words (ViT Paper)](https://arxiv.org/abs/2010.11929)
