"""
Train Orion 1B from scratch on RTX 3060 Ti (8GB VRAM).
- Resume from checkpoint
- Gradient checkpointing to fit in VRAM
- Frequent saves for pause/resume
- Designed to run overnight
"""
import os, sys, time, math, json, csv, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "C:/Users/avery/orion-from-scratch")
from model import OrionModel, OrionConfig, CONFIGS

DATA_DIR = "C:/Users/avery/orion-from-scratch/data"
CKPT_DIR = "C:/Users/avery/orion-from-scratch/checkpoints"
LOG_DIR = "C:/Users/avery/orion-from-scratch/logs"
STATUS_FILE = os.path.join(LOG_DIR, "train_1B_status.json")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def load_data(split, block_size, batch_size, device):
    # Try v3 first, then v2, then v1
    for ver in ["v3", "v2", ""]:
        suffix = f"_{ver}" if ver else ""
        path = os.path.join(DATA_DIR, f"{split}{suffix}.bin")
        if os.path.exists(path):
            break
    print(f"  Data: {path} ({os.path.getsize(path)/1e9:.2f}GB)")
    data = np.memmap(path, dtype=np.uint16, mode='r')
    
    def get_batch():
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)
    return get_batch

@torch.no_grad()
def estimate_loss(model, get_train, get_val, iters=50):
    model.eval()
    losses = {}
    for name, fn in [("train", get_train), ("val", get_val)]:
        total = 0.0
        for _ in range(iters):
            x, y = fn()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
            total += loss.item()
        losses[name] = total / iters
    model.train()
    return losses

def save_checkpoint(model, optimizer, scaler, config, iteration, best_val, path):
    state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
    torch.save({
        'model': state,
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler else None,
        'config': config,
        'iter': iteration,
        'best_val_loss': best_val,
    }, path)

def update_status(data):
    with open(STATUS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--max_iters", type=int, default=50000)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()
    
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # 1B config - fits in 8GB with gradient checkpointing
    config = CONFIGS['1B']
    config.max_seq_len = args.block_size
    config.dropout = 0.0
    config.vocab_size = 32000
    
    start_iter = 0
    best_val_loss = float('inf')
    
    model = OrionModel(config).to(device)
    
    # Enable gradient checkpointing to save VRAM
    # Manually wrap each layer's forward
    original_forwards = []
    for layer in model.layers:
        original_forwards.append(layer.forward)
        def make_ckpt_forward(orig_fn):
            def ckpt_forward(x, cos, sin, mask=None):
                return torch.utils.checkpoint.checkpoint(orig_fn, x, cos, sin, mask, use_reentrant=False)
            return ckpt_forward
        layer.forward = make_ckpt_forward(layer.forward)
    
    params = model.count_params()
    print(f"\nOrion 1B: {params:,} parameters ({params/1e6:.0f}M)")
    print(f"Config: dim={config.dim} layers={config.n_layers} heads={config.n_heads}")
    print(f"Block: {args.block_size} | Batch: {args.batch_size} | Grad accum: {args.grad_accum}")
    print(f"Effective batch: {args.batch_size * args.grad_accum * args.block_size / 1e6:.1f}M tokens")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    scaler = torch.amp.GradScaler('cuda')
    
    # Resume from checkpoint
    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model_state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        # Load model weights
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt and ckpt['optimizer']:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt and ckpt['scaler']:
            scaler.load_state_dict(ckpt['scaler'])
        start_iter = ckpt.get('iter', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed at iter {start_iter}, best_val={best_val_loss:.4f}")
    
    # Try torch.compile
    try:
        # model = torch.compile(model)  # Triton not available on Windows
        print('torch.compile skipped (Windows)')
    except:
        print("torch.compile not available")
    
    get_train = load_data("train", args.block_size, args.batch_size, device)
    get_val = load_data("val", args.block_size, args.batch_size, device)
    
    def get_lr(it):
        if it < args.warmup_iters:
            return args.lr * (it + 1) / args.warmup_iters
        if it > args.max_iters:
            return args.lr * 0.1
        ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        return args.lr * 0.1 + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (args.lr - args.lr * 0.1)
    
    log_file = os.path.join(LOG_DIR, f"train_1B_{int(time.time())}.csv")
    csv_f = open(log_file, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["iter", "train_loss", "val_loss", "lr", "ms", "tok_s"])
    
    print(f"\nTraining from iter {start_iter} to {args.max_iters}")
    print(f"Save every {args.save_interval}, eval every {args.eval_interval}")
    print("=" * 60)
    
    model.train()
    t0 = time.time()
    
    for it in range(start_iter, args.max_iters):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for micro in range(args.grad_accum):
            x, y = get_train()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        tps = args.batch_size * args.block_size * args.grad_accum / dt if dt > 0 else 0
        
        if it % args.log_interval == 0:
            vram = torch.cuda.max_memory_allocated() / 1e9
            print(f"iter {it:6d} | loss {accum_loss:.4f} | lr {lr:.2e} | {dt*1000:.0f}ms | {tps:.0f} tok/s | {vram:.1f}GB VRAM")
            update_status({
                "iter": it, "max_iters": args.max_iters, "loss": round(accum_loss, 4),
                "lr": lr, "tok_s": round(tps), "vram_gb": round(vram, 1),
                "pct": round(it / args.max_iters * 100, 1),
                "updated": time.strftime('%Y-%m-%d %H:%M:%S'),
            })
        
        if it % args.eval_interval == 0 and it > 0:
            losses = estimate_loss(model, get_train, get_val)
            print(f"  >>> EVAL {it}: train={losses['train']:.4f} val={losses['val']:.4f}")
            writer.writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.6f}", f"{dt*1000:.0f}", f"{tps:.0f}"])
            csv_f.flush()
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_checkpoint(model, optimizer, scaler, config, it, best_val_loss,
                    os.path.join(CKPT_DIR, "orion_1B_best.pt"))
                print(f"  >>> New best! val_loss={best_val_loss:.4f}")
        
        if it % args.save_interval == 0 and it > 0:
            save_checkpoint(model, optimizer, scaler, config, it, best_val_loss,
                os.path.join(CKPT_DIR, f"orion_1B_iter{it}.pt"))
            # Keep only last 3 periodic checkpoints to save disk
            ckpts = sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("orion_1B_iter")], 
                          key=lambda x: int(x.split("iter")[1].split(".")[0]))
            while len(ckpts) > 3:
                os.remove(os.path.join(CKPT_DIR, ckpts.pop(0)))
            print(f"  >>> Checkpoint saved at iter {it}")
    
    save_checkpoint(model, optimizer, scaler, config, args.max_iters, best_val_loss,
        os.path.join(CKPT_DIR, "orion_1B_final.pt"))
    print(f"\nDone! Best val loss: {best_val_loss:.4f}")
    csv_f.close()

if __name__ == "__main__":
    train()



