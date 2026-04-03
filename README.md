# Structura

`Structura` e um workspace Rust para visao computacional com foco em:

- geometria multi-view
- extracao e matching de features
- calibracao de camera
- bundle adjustment e etapas de Structure from Motion (SfM)

## Pre-requisitos

- Rust recente com suporte a `edition = "2024"`
- `cargo`
- bibliotecas nativas exigidas pelas dependencias, principalmente OpenCV e ONNX Runtime
- modelos ONNX na pasta `models/`

Observacao: alguns testes e wrappers ONNX podem depender de providers especificos de inferencia. O projeto ja contem suporte condicional por variaveis de ambiente como `STRUCTURA_ORT_ROCM` e `STRUCTURA_ORT_MIGRAPHX`.

## Como executar

Hoje o projeto nao expoe um binario principal em `crates/cli`, entao o fluxo pratico e via `cargo test` e consumo dos crates como biblioteca.

Compilar o workspace:

```bash
cargo check
```

Rodar todos os testes:

```bash
cargo test
```

Rodar apenas os testes de features:

```bash
cargo test -p structura-feature
```

Rodar apenas os testes de SfM:

```bash
cargo test -p structura-sfm
```

## Modelos e artefatos

- os modelos ONNX ficam em `models/`
- alguns testes de `lightglue_onnx` geram artefatos visuais em `target/lightglue-onnx-tests/`

## Estado atual

O workspace esta estruturado principalmente como base de bibliotecas e testes. Se voce quiser adicionar uma interface executavel, o lugar natural para isso e `crates/cli`.
