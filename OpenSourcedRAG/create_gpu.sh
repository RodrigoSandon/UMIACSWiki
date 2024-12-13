#!/bin/bash

# Define the machine type, GPU type, and snapshot details
MACHINE_TYPE="n1-standard-8"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
SNAPSHOT_NAME="qdrant-llama-rag-pipeline-ready"

echo "Starting the creation of GPU instances..."

for zone in $(gcloud compute accelerator-types list --filter="name:$GPU_TYPE" --format="value(zone)" | awk '{lines[NR] = $0} END {for (i = NR; i > 0; i--) print lines[i]}'); do
    echo "Attempting to create a VM instance in zone: $zone"
    if gcloud compute instances create "gpu-machine-$zone" \
        --zone="$zone" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --maintenance-policy=TERMINATE \
        --restart-on-failure \
        --boot-disk-size="100GB" \
        --boot-disk-type="pd-standard" \
        --boot-disk-device-name="gpu-machine-$zone-disk" \
        --source-snapshot="$SNAPSHOT_NAME" \
        --metadata="install-nvidia-driver=True"; then
        echo "VM instance successfully created in zone: $zone"
        break  
    else
        echo "Failed to create VM instance in zone: $zone. Trying next zone..."
    fi
done

echo "Finished attempting to create GPU instances."