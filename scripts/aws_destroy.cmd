@echo off
cd infra\terraform
echo Run:
echo terraform destroy -var="key_name=YOUR_KEY" -var="private_key_path=C:/path/key.pem"
