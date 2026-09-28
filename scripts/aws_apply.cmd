@echo off
cd infra\terraform
terraform init
echo Run:
echo terraform apply -var="key_name=YOUR_KEY" -var="private_key_path=C:/path/key.pem"
