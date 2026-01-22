 Rebuild with the fixed code
docker build -t homomorphic-auth .

# Then run:
docker run --rm homomorphic-auth python run_benchmark.py