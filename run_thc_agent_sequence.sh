#!/bin/bash

# Function to run a command multiple times
run_multiple_times() {
    local command="$1"
    local times="$2"
    local action="$3"
    
    echo "Running $action $times times..."
    for i in $(seq 1 $times); do
        echo "Iteration $i of $times"
        $command
        echo "----------------------------------------"
    done
}

# Run find_company_details 10 times
run_multiple_times "uv run src/run_thc_agent.py --action find_company_details" 10 "find_company_details"

# Run find_products 10 times
run_multiple_times "uv run src/run_thc_agent.py --action find_products" 5 "find_products"

# Run set_product_flavors once
echo "Running set_product_flavors..."
uv run src/run_thc_agent.py --action set_product_flavors
echo "----------------------------------------"

# Run set_product_effect once
echo "Running set_product_effect..."
uv run src/run_thc_agent.py --action set_product_effect
echo "----------------------------------------" 