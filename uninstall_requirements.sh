#!/bin/bash

# Check if uninstall_requirements.txt exists
if [ ! -f uninstall_requirements.txt ]; then
    echo "uninstall_requirements.txt not found!"
    exit 1
fi

# Read each line in uninstall_requirements.txt and uninstall the package
while IFS= read -r package || [ -n "$package" ]; do
    # Remove version specification for pip uninstall
    package_name=$(echo "$package" | sed 's/[><=].*//')
    echo "Uninstalling $package_name"
    pip uninstall -y "$package_name"
done < uninstall_requirements.txt
