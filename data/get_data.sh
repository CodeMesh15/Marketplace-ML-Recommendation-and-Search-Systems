
# This script provides instructions for downloading the Yelp Dataset
# needed for the Marketplace-ML project.

# Create the directory for the raw data
mkdir -p raw_data

echo "--- Marketplace-ML Data Download Guide ---"
echo ""
echo "This script will not download the data automatically due to website terms."
echo "Please follow these manual steps:"
echo ""
echo "1. Open a web browser and go to the Yelp Open Dataset page:"
echo "   https://www.yelp.com/dataset"
echo ""
echo "2. Click 'Download Dataset' and agree to the terms to download the files."
echo ""
echo "3. Once downloaded, extract the archive. You will find several JSON files."
echo ""
echo "4. Move the following essential JSON files into the 'data/raw_data/' directory"
echo "   that has been created in this project:"
echo "   - yelp_academic_dataset_business.json"
echo "   - yelp_academic_dataset_review.json"
echo ""
echo "After these files are in place, you can run the preprocess.py script."
echo ""
echo "------------------------------------------"
