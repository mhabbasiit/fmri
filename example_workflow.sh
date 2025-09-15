#!/bin/bash

# fMRI Pipeline - Example Workflow
# This script demonstrates a complete pipeline run from setup to connectivity analysis

echo "🧠 fMRI Pipeline - Example Workflow"
echo "=================================="

# Step 1: Check prerequisites
echo "📋 Step 1: Checking prerequisites..."
./launcher.sh --check
if [ $? -ne 0 ]; then
    echo "❌ Prerequisites check failed. Please fix issues before continuing."
    exit 1
fi
echo "✅ Prerequisites check passed!"
echo ""

# Step 2: Discover subjects
echo "🔍 Step 2: Discovering subjects in BIDS directory..."
python subject_manager.py --action discover --output discovered_subjects.txt
echo "✅ Subjects discovered and saved to discovered_subjects.txt"
echo ""

# Step 3: Test single subject
echo "🧪 Step 3: Testing single subject processing..."
FIRST_SUBJECT=$(head -n 1 discovered_subjects.txt | sed 's/sub-//')
echo "Testing with subject: $FIRST_SUBJECT"
python pipeline_runner.py --mode single --subject $FIRST_SUBJECT --cpu-only
if [ $? -eq 0 ]; then
    echo "✅ Single subject test successful!"
else
    echo "❌ Single subject test failed. Check logs for details."
    exit 1
fi
echo ""

# Step 4: Parallel processing (first 2 subjects)
echo "⚡ Step 4: Running parallel processing for 2 subjects..."
SECOND_SUBJECT=$(sed -n '2p' discovered_subjects.txt | sed 's/sub-//')
if [ ! -z "$SECOND_SUBJECT" ]; then
    echo "Processing subjects: $FIRST_SUBJECT and $SECOND_SUBJECT"
    python pipeline_runner.py --mode parallel --subjects $FIRST_SUBJECT $SECOND_SUBJECT --cpu-only
    echo "✅ Parallel processing completed!"
else
    echo "⚠️ Only one subject available, skipping parallel test"
fi
echo ""

# Step 5: Generate connectivity matrices
echo "🕸️ Step 5: Generating connectivity matrices..."
python run_connectivity.py --all-completed --atlases aal harvard_oxford
echo "✅ Connectivity matrices generated!"
echo ""

# Step 6: Final status check
echo "📊 Step 6: Final pipeline status..."
python pipeline_runner.py --mode status
echo ""

# Step 7: Show output structure
echo "📁 Step 7: Output directory structure..."
echo "Main outputs:"
find ADNI-FMRI/FMRI/ -name "*.nii.gz" -type f | head -5
echo ""
echo "Connectivity matrices:"
find ADNI-FMRI/connectivity_matrices/ -name "*.csv" -type f | head -5
echo ""
echo "QC reports:"
find ADNI-FMRI/FMRI/ -name "*.html" -type f | head -3
echo ""

echo "🎉 Workflow completed successfully!"
echo ""
echo "Next steps:"
echo "- Review QC reports in ADNI-FMRI/FMRI/sub-*/figures/"
echo "- Check connectivity matrices in ADNI-FMRI/connectivity_matrices/"
echo "- Use 'python pipeline_runner.py --mode status' to monitor ongoing processes"
echo "- Process remaining subjects with 'python pipeline_runner.py --mode watch'"
