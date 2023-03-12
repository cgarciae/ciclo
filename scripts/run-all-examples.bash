set -e
poetry shell

for f in $(find examples -name "*.py"); do
    echo -e "\n---------------------------------"
    echo "$f"
    echo "---------------------------------"
    python "$f"
done
