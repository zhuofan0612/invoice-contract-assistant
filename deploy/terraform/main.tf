/*
 * Infrastructure for the invoice-contract assistant.
 *
 * Scope note, because it matters for how this should be read: this is a
 * personal project and the module below has never been applied against a real
 * cloud account. It is written to show what the deployment *is* -- a managed
 * Postgres with pgvector, a private network, secrets held outside the image --
 * not to claim production operations experience.
 *
 * The parts that reflect real constraints rather than defaults:
 *
 *   - `data_residency` is a variable and the region is validated against EU
 *     regions. The sovereignty requirement is the reason this system can be
 *     switched to a self-hosted model, and it applies to stored data too: a
 *     municipality's procurement records should not silently land elsewhere.
 *   - The database is not publicly accessible and lives in private subnets.
 *   - `deletion_protection` defaults to true. Losing the clause index is
 *     recoverable by re-ingesting; losing the decision traces is not, and they
 *     are the audit record for decisions about public money.
 */

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "region" {
  description = "Deployment region. Constrained to the EU for data residency."
  type        = string
  default     = "eu-north-1" # Stockholm

  validation {
    condition     = startswith(var.region, "eu-")
    error_message = "Procurement records must stay in the EU; region must start with 'eu-'."
  }
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.medium"
}

variable "deletion_protection" {
  description = "Traces are the audit record for decisions about public money."
  type        = bool
  default     = true
}

provider "aws" {
  region = var.region
}

locals {
  name = "invoice-check-${var.environment}"
  tags = {
    Project     = "invoice-check"
    Environment = var.environment
    ManagedBy   = "terraform"
    DataClass   = "procurement-restricted"
  }
}

# --- network --------------------------------------------------------------

resource "aws_vpc" "main" {
  cidr_block           = "10.40.0.0/16"
  enable_dns_hostnames = true
  tags                 = merge(local.tags, { Name = local.name })
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags              = merge(local.tags, { Name = "${local.name}-private-${count.index}" })
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_db_subnet_group" "main" {
  name       = local.name
  subnet_ids = aws_subnet.private[*].id
  tags       = local.tags
}

resource "aws_security_group" "db" {
  name        = "${local.name}-db"
  description = "Postgres reachable only from inside the VPC"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

# --- database -------------------------------------------------------------

resource "aws_db_instance" "main" {
  identifier     = local.name
  engine         = "postgres"
  engine_version = "16"
  instance_class = var.db_instance_class

  allocated_storage     = 50
  max_allocated_storage = 200
  storage_encrypted     = true

  db_name  = "invoice_check"
  username = "invoice"
  # Rotated by Secrets Manager rather than held in state.
  manage_master_user_password = true

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible    = false

  backup_retention_period = 14
  deletion_protection     = var.deletion_protection
  skip_final_snapshot     = false
  final_snapshot_identifier = "${local.name}-final"

  # pgvector is an extension, so it has to be allow-listed before the schema's
  # CREATE EXTENSION will succeed.
  parameter_group_name = aws_db_parameter_group.main.name

  tags = local.tags
}

resource "aws_db_parameter_group" "main" {
  name   = local.name
  family = "postgres16"

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
    apply_method = "pending-reboot"
  }

  tags = local.tags
}

# --- secrets --------------------------------------------------------------

resource "aws_secretsmanager_secret" "llm" {
  name        = "${local.name}/llm-api-key"
  description = "Model API key. Empty is valid: the service falls back to the deterministic client."
  tags        = local.tags
}

output "db_endpoint" {
  value       = aws_db_instance.main.endpoint
  description = "Feed into the DATABASE_URL secret the Helm chart expects."
}

output "llm_secret_arn" {
  value = aws_secretsmanager_secret.llm.arn
}
