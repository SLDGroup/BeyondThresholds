#!/bin/bash

# Navigate to the parent folder
cd rwhar/data

# Loop through each subfolder
for subfolder in */; do
    # Navigate to the subfolder
    cd "$subfolder"
    rm -rf images videos
    cd "data"
    
    # Loop through each zip file
    for zip_file in *.zip; do
        # Check if the zip file starts with "acc_"
        if [[ "$zip_file" == acc_*_csv.zip ]]; then
            # Unzip the file
            unzip "$zip_file"
            rm readMe
        fi
            
        # Remove the file
        rm "$zip_file"
        
    done

    # Loop through each zip file (some folders have nested zip files)
    for zip_file in *.zip; do
        # Check if the zip file starts with "acc_"
        if [[ "$zip_file" == acc_*_csv.zip ]]; then
            # Unzip the file
            unzip "$zip_file"
            rm readMe
        fi
            
        # Remove the file
        rm "$zip_file"
        
    done
    
    # Navigate back to the parent folder
    cd ../..
done
