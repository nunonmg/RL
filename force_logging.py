#!/usr/bin/env python3
"""
Force logging configuration for NeMo RL training
Run this before your training to ensure all prints are visible
"""

import sys
import os
import logging
from datetime import datetime

def setup_multinode_logging():
    """
    Configure logging to ensure visibility in multi-node setups
    """
    # Force stdout/stderr to be unbuffered
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    # Set environment variables for unbuffered output
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONFAULTHANDLER'] = '1'
    
    # Configure logging with node information
    node_rank = os.environ.get('NODE_RANK', 'unknown')
    log_format = f'[NODE_{node_rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'/mnt/nunomg/nemo-rl/python_debug_node_{node_rank}.log')
        ]
    )
    
    # Override print function to add node info
    original_print = print
    def enhanced_print(*args, **kwargs):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        node_info = f"[NODE_{node_rank}][{timestamp}]"
        original_print(node_info, *args, **kwargs)
        # Also force flush
        sys.stdout.flush()
    
    # Replace built-in print (be careful with this approach)
    import builtins
    builtins.print = enhanced_print
    
    # Force all loggers to be visible
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler(sys.stdout))
    
    print("🔧 Enhanced logging configured for multi-node training")
    print(f"📍 Node rank: {node_rank}")
    print(f"📊 Python unbuffered: {os.environ.get('PYTHONUNBUFFERED')}")

if __name__ == "__main__":
    setup_multinode_logging()
    print("✅ Logging setup complete")