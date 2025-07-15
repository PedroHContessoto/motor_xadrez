# GPU Support Setup

## Compilação Padrão (CPU Only)
```bash
cargo run --release demo
cargo run --release train
cargo run --release benchmark
```

## Compilação com GPU (Requires CUDA Toolkit)

### Pré-requisitos Windows:
1. **Visual Studio Build Tools**: Instalar "C++ build tools"
2. **CUDA Toolkit**: Versão 11.8+ 
3. **NVCC**: Deve estar no PATH

### Pré-requisitos Linux:
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit build-essential

# Arch Linux  
sudo pacman -S cuda gcc
```

### Compilação com CUDA:
```bash
cargo run --release --features cuda demo
cargo run --release --features cuda train
cargo run --release --features cuda benchmark
```

## Performance Esperada

### CPU (Baseline):
- Single evaluation: ~150 evals/sec
- Batch evaluation: ~1,200 evals/sec
- Training: ~5 min/iteration

### GPU (CUDA):
- Single evaluation: ~400 evals/sec  
- Batch evaluation: ~15,000 evals/sec
- Training: ~1 min/iteration

## Configuração Manual

```rust
use motor_xadrez::{NNUEConfig, SelfPlayConfig};

// Force CPU
let config = NNUEConfig {
    use_gpu: false,
    precision: "f32".to_string(),
    batch_size: 32,
    device_id: 0,
};

// Force GPU (if available)
let config = NNUEConfig {
    use_gpu: true,
    precision: "f16".to_string(), // Mixed precision
    batch_size: 256,
    device_id: 0,
};
```

## Troubleshooting

### Erro "Cannot find compiler 'cl.exe'":
- Instalar Visual Studio Build Tools
- Ou usar: `cargo run --release` (sem --features cuda)

### Erro "CUDA not found":
- Verificar CUDA_PATH ambiente
- Ou usar: `cargo run --release` (CPU fallback)

### Performance baixa:
- Verificar GPU sendo usada: logs mostram "device=Cuda(0)"
- Verificar batch_size adequado para GPU memory