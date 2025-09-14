#!/usr/bin/env python3
"""
Training Monitor Script
Monitors GPU usage and training progress in real-time
"""

import time
import subprocess
import threading
import sys

def monitor_gpu():
    """Monitor GPU usage continuously"""
    print("🖥️  GPU Monitoring Started (Press Ctrl+C to stop)")
    print("=" * 80)
    
    try:
        while True:
            # Get GPU info
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                name = gpu_info[0]
                temp = gpu_info[1]
                gpu_util = gpu_info[2]
                mem_util = gpu_info[3]
                mem_used = gpu_info[4]
                mem_total = gpu_info[5]
                power = gpu_info[6]
                
                # Clear line and print GPU stats
                print(f"\r🔥 {name[:20]} | {temp}°C | GPU: {gpu_util}% | MEM: {mem_used}/{mem_total}MB ({mem_util}%) | PWR: {power}W", end="", flush=True)
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n🛑 GPU monitoring stopped")

def monitor_processes():
    """Monitor GPU processes"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-compute-apps=pid,process_name,used_gpu_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            print("\n\n📋 GPU Processes:")
            print("-" * 50)
            processes = result.stdout.strip().split('\n')
            for proc in processes:
                if 'python' in proc.lower():
                    pid, name, mem = proc.split(', ')
                    print(f"🐍 PID: {pid} | {name} | Memory: {mem}MB")
        else:
            print("\n\n⚠️  No GPU processes found")
            
    except Exception as e:
        print(f"\n\n❌ Error checking processes: {e}")

def show_training_tips():
    """Show tips for monitoring training"""
    print("\n" + "=" * 80)
    print("📚 TRAINING MONITORING TIPS")
    print("=" * 80)
    print("✅ GOOD SIGNS:")
    print("   • GPU Utilization: 70-95%")
    print("   • Memory Usage: 8-15GB (your training)")
    print("   • Temperature: 50-80°C")
    print("   • Loss decreasing over epochs")
    print("   • Process name shows 'python'")
    
    print("\n⚠️  WARNING SIGNS:")
    print("   • GPU Utilization: <50% (data loading bottleneck)")
    print("   • Memory Usage: <2GB (not using GPU properly)")
    print("   • Temperature: >85°C (overheating)")
    print("   • Loss not decreasing (learning issues)")
    
    print("\n🔧 COMMANDS TO RUN IN SEPARATE TERMINALS:")
    print("   • nvidia-smi -l 2        (continuous GPU monitoring)")
    print("   • htop                   (CPU/RAM monitoring)")
    print("   • python monitor_training.py  (this script)")
    
    print("\n📊 EXPECTED PERFORMANCE (RTX 5090):")
    print("   • Batch Time: 2-5 seconds")
    print("   • Epoch Time: 10-15 minutes") 
    print("   • GPU Memory: 8-12GB")
    print("   • Total Training: ~70 minutes")
    print("=" * 80)

def main():
    """Main monitoring function"""
    print("🚀 Kronos QQQ Training Monitor")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--gpu-only':
        monitor_gpu()
        return
    
    # Show tips first
    show_training_tips()
    
    # Check current processes
    monitor_processes()
    
    print("\n\n🔄 Starting real-time GPU monitoring...")
    print("   (Run this in a separate terminal while training)")
    
    # Start GPU monitoring
    monitor_gpu()

if __name__ == "__main__":
    main()