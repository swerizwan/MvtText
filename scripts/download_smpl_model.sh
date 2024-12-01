mkdir -p reqs/
cd reqs/

echo "The smpl model will be stored in the './reqs' folder"

# HumanAct12 poses
echo "Downloading"
gdown "https://drive.google.com/uc?id=1qrFkPZyRwRGd0Q3EY76K8oJaIgs_WK9i"
echo "Extracting"
tar xfzv smpl.tar.gz
echo "Cleaning"
rm smpl.tar.gz

echo "Downloading done!"
