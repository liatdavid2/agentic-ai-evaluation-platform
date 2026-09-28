variable "aws_region" { type = string, default = "eu-central-1" }
variable "instance_type" { type = string, default = "t3.small" }
variable "key_name" { type = string, description = "Existing EC2 key pair name" }
variable "private_key_path" { type = string, description = "Local path to private PEM key" }
variable "allowed_cidr" {
  type = string
  description = "CIDR allowed to access demo services"
  default = "0.0.0.0/0"
}
