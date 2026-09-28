output "public_ip" { value = aws_instance.agent_eval.public_ip }
output "ui_url" { value = "http://${aws_instance.agent_eval.public_ip}:8080" }
output "api_docs_url" { value = "http://${aws_instance.agent_eval.public_ip}:8000/docs" }
