provider "aws" { region = var.aws_region }

data "aws_ami" "ubuntu" {
  most_recent = true
  owners = ["099720109477"]
  filter {
    name = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
}

resource "aws_security_group" "agent_eval" {
  name_prefix = "agent-eval-"
  ingress { description="SSH"; from_port=22; to_port=22; protocol="tcp"; cidr_blocks=[var.allowed_cidr] }
  ingress { description="React UI"; from_port=8080; to_port=8080; protocol="tcp"; cidr_blocks=[var.allowed_cidr] }
  ingress { description="FastAPI"; from_port=8000; to_port=8000; protocol="tcp"; cidr_blocks=[var.allowed_cidr] }
  egress { from_port=0; to_port=0; protocol="-1"; cidr_blocks=["0.0.0.0/0"] }
}

resource "aws_instance" "agent_eval" {
  ami = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  key_name = var.key_name
  vpc_security_group_ids = [aws_security_group.agent_eval.id]
  root_block_device { volume_size=20; volume_type="gp3" }

  user_data = <<-EOF
    #!/bin/bash
    set -eux
    apt-get update
    apt-get install -y docker.io docker-compose-v2
    systemctl enable --now docker
    usermod -aG docker ubuntu
  EOF

  tags = { Name = "agentic-ai-evaluation-platform" }
}

resource "null_resource" "deploy" {
  depends_on = [aws_instance.agent_eval]
  triggers = { instance_id = aws_instance.agent_eval.id }

  connection {
    type = "ssh"
    host = aws_instance.agent_eval.public_ip
    user = "ubuntu"
    private_key = file(var.private_key_path)
    timeout = "8m"
  }

  provisioner "remote-exec" {
    inline = ["mkdir -p /home/ubuntu/agentic-ai-evaluation-platform"]
  }

  provisioner "file" {
    source = "../../"
    destination = "/home/ubuntu/agentic-ai-evaluation-platform"
  }

  provisioner "remote-exec" {
    inline = [
      "cd /home/ubuntu/agentic-ai-evaluation-platform",
      "cp -n .env.example .env || true",
      "sudo docker compose up -d --build"
    ]
  }
}
