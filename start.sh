#!/bin/bash
set -e

echo " Starting GraphPlag API on Railway..."
echo "PORT: $PORT"
echo "Environment: ${RAILWAY_ENVIRONMENT:-development}"

# Apply GraKeL patches if needed
if [ ! -f ".grakel_patched" ]; then
    echo " Applying GraKeL patches..."
    python -c "
import grakel
from pathlib import Path

# Patch RandomWalk kernel
rw_file = Path(grakel.__file__).parent / 'kernels' / 'randomwalk.py'
content = rw_file.read_text()
if 'p = p * M' in content:
    content = content.replace('p = p * M', 'p = np.dot(p, M)')
    rw_file.write_text(content)
    print('✓ RandomWalk kernel patched')

# Patch ShortestPath kernel
sp_file = Path(grakel.__file__).parent / 'kernels' / 'shortest_path.py'
content = sp_file.read_text()
if 'K[i, j] += count_i.get(label, 0) * count_j.get(label, 0)' in content:
    content = content.replace(
        'K[i, j] += count_i.get(label, 0) * count_j.get(label, 0)',
        'if label in count_i and label in count_j:\\n                    K[i, j] += count_i[label] * count_j[label]'
    )
    sp_file.write_text(content)
    print('✓ ShortestPath kernel patched')
" || echo "⚠️ Patches already applied or not needed"
    
    touch .grakel_patched
fi

# Create cache directory
mkdir -p .cache

# Start the API
echo " Starting Uvicorn server..."
exec python -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
