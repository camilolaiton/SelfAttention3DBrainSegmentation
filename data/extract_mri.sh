#!/bin/bash

# Made by:
#         - Camilo Laiton
#         Universidad Nacional de Colombia, Colombia
#         2021-1
#         GitHub: https://github.com/camilolaiton/

#         This file belongs to the private repository "master_thesis" where
#         I save all the files that are related to my thesis which is called
#         "Método para la segmentación de imágenes de resonancia magnética 
#         cerebrales usando una arquitectura de red neuronal basada en modelos
#         de atención".

# Usage example: $ bash extract_mri.sh ../other_directory/MMRR-21/
# This bash script will extract the files listed below, create the respective folders
# and put the extracted files where they belong.

# Files:
#   - 001.mgz
#   - aparcNMMjt+aseg.mgz

# Example: From MMRR-21-1.tar.gz this script creates the folder MMRR-21-1 and puts 001.mgz and
# aparcNMMjt+aseg.mgz that belongs to MMRR-21-1


orig_path="Users/arno.klein/Data/Mindboggle101/subjects"
orig_file="mri/orig/001.mgz"
strip_orig=8

segmented_file="mri/aparcNMMjt+aseg.mgz"
strip_seg=7

# echo "Root of tar files: $1"

if [ -z "$1" ] 
  then
    echo "Please, give me a root folder where tar.gz mri brain scans are."
    exit 1
fi

for file in $1*
do
    foldername=$(basename "$file" .tar.gz)
    mkdir "$1$foldername"
    
    # orig file
    cmd="-zxf $file $orig_path/$foldername/$orig_file --directory "$1$foldername" --strip-components $strip_orig"
    tar $cmd
    mv $(basename $orig_file) "$1$foldername"

    # segmented file
    cmd="-zxf $file $orig_path/$foldername/$segmented_file --directory "$1$foldername" --strip-components $strip_seg"
    tar $cmd
    mv $(basename $segmented_file) "$1$foldername"
done

# Remove all extracted files
rm -r $1*.tar.gz
# Move the extracted files
mv $1 /home/camilo/Programacion/master_thesis/data