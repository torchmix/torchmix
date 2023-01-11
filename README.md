<h1 align="center">torchmix</h1>

<h3 align="center">The missing component library for PyTorch</h3>

<br />

`torchmix` is a collection of PyTorch modules that aims to simplify your model development process with pre-made PyTorch components. We've included a range of operations, from basic ones like `Repeat` and `Add`, to more complex ones like `WindowAttention` in the [Swin-Transformer](https://arxiv.org/abs/2103.14030). Our goal is to make it easy for you to use these various operations with minimal code, so you can focus on building your project rather than writing boilerplate.

We've designed `torchmix` to be as user-friendly as possible. Each implementation is kept minimal and easy to understand, using [`einops`](https://github.com/arogozhnikov/einops) to avoid confusing tensor manipulation (such as `permute`, `transpose`, and `reshape`) and [`jaxtyping`](https://github.com/google/jaxtyping) to clearly document the shapes of the input and output tensors. This means that you can use `torchmix` with confidence, knowing that the components you're working with are clean and reliable.

**Note: `torchmix` is a prototype that is currently in development and has not been tested for production use. The API may change at any time.**

## Install

To use `torchmix`, you will need to have `torch` already installed on your environment.

```sh
pip install torchmix
```

## Documentation

To learn more, check out our [documentation](https://torchmix.vercel.app).

## Contributing

The development of `torchmix` is an open process, and we welcome any contributions or suggestions for improvement. If you have ideas for new components or ways to enhance the library, feel free to open an issue or start a discussion. We welcome all forms of feedback, including criticism and suggestions for significant design changes. Please note that `torchmix` is currently in the early stages of development and any contributions should be considered experimental. Thank you for your support of `torchmix`!

## License

`torchmix` is licensed under the MIT License.
