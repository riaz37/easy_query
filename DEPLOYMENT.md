# Deployment Guide

This guide explains how to deploy the Easy Query Knowledge Base application to different environments.

## Docker Deployment (Recommended)

The easiest way to deploy Easy Query is using Docker and Docker Compose with the provided configuration.

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 1.29+
- Python 3.11+ (for local development)
- Node.js 18+ (for local development)
- pnpm (preferred package manager)

### Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd easy_query
```

2. Configure environment variables:
```bash
cp backend/.env.example backend/.env
# Edit backend/.env with your configuration
```

3. Start all services:
```bash
docker-compose up -d
```

4. Access the application:
- Frontend: http://localhost:3000
- Backend API: https://localhost:8200
- Database: localhost:5432
- Redis: localhost:6379

### Stopping Services

```bash
docker-compose down
```

To stop and remove volumes (data will be lost):
```bash
docker-compose down -v
```

## Manual Deployment

### Backend Deployment

1. Install Python 3.11+ and create a virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
# Create .env file with production values
GOOGLE_API_KEY=your_production_google_api_key
GROQ_API_KEY=your_production_groq_api_key
OPENAI_API_KEY=your_production_openai_api_key

# Database Configuration
DB_HOST=your_production_db_host
DB_PORT=5432
DB_USER=your_production_db_user
DB_PASSWORD=your_production_db_password
DB_NAME=your_production_db_name

# Redis Configuration (optional)
REDIS_CLOUD=true
REDIS_HOST=your_redis_host
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Server Configuration
ENVIRONMENT=production
PROD_BACKEND_URL=https://your-production-domain.com:8200

# Authentication
SECRET_KEY=your_secure_secret_key_change_this_in_production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

4. Run the application:
```bash
# For production (HTTPS)
python main.py

# For development (HTTP)
python main.py --http
```

For production, use a process manager like PM2 or systemd:
```bash
# Using PM2
pm2 start main.py --name "easy-query-backend"

# Using systemd (create /etc/systemd/system/easy-query.service)
[Unit]
Description=Easy Query Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/easy_query/backend
Environment=PATH=/path/to/easy_query/backend/venv/bin
ExecStart=/path/to/easy_query/backend/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Frontend Deployment

1. Install Node.js 18+ and dependencies:
```bash
cd frontend
pnpm install
```

2. Build the application:
```bash
pnpm build
```

3. Start the production server:
```bash
pnpm start
```

For production, use a reverse proxy like Nginx or Apache.

### Nginx Configuration Example

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api/ {
        proxy_pass https://localhost:8200/;
        proxy_ssl_verify off;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support for voice agent
    location /voice/ws {
        proxy_pass https://localhost:8200;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_ssl_verify off;
    }
}
```

## Cloud Deployment

### AWS Deployment

1. **Elastic Beanstalk**:
   - Create a new application
   - Upload the backend as a ZIP file
   - Configure environment variables in the console

2. **EC2**:
   - Launch an EC2 instance
   - Install Docker and Docker Compose
   - Follow the Docker deployment steps

3. **RDS**:
   - Create a PostgreSQL RDS instance
   - Update database connection details in `.env`

4. **Elastic Container Service (ECS)**:
   - Create task definitions for each service
   - Deploy containers to ECS cluster

### Google Cloud Platform

1. **Cloud Run**:
   - Build and push Docker images to Container Registry
   - Deploy each service to Cloud Run

2. **Compute Engine**:
   - Launch a VM instance
   - Install Docker and Docker Compose
   - Follow the Docker deployment steps

3. **Cloud SQL**:
   - Create a PostgreSQL instance
   - Update database connection details in `.env`

### Azure

1. **Azure Container Instances**:
   - Deploy Docker containers directly

2. **Azure App Service**:
   - Deploy frontend as a web app
   - Deploy backend as an API app

3. **Azure Database for PostgreSQL**:
   - Create a PostgreSQL database
   - Update database connection details in `.env`

## Environment Variables for Production

Ensure these variables are set in production:

```env
# ===========================================
# AI MODEL CONFIGURATION
# ===========================================
GOOGLE_API_KEY=your_production_google_api_key
GROQ_API_KEY=your_production_groq_api_key
OPENAI_API_KEY=your_production_openai_api_key

# Model Selection
GOOGLE_GEMINI_NAME=gemini-1.5-pro
GOOGLE_GEMINI_EMBEDDING_NAME=gemini-embedding-exp-03-07

# ===========================================
# DATABASE CONFIGURATION
# ===========================================
# PostgreSQL
DB_HOST=your_production_db_host
DB_PORT=5432
DB_USER=your_production_db_user
DB_PASSWORD=your_production_db_password
DB_NAME=your_production_db_name

# Microsoft SQL Server (Alternative)
MSSQL_SERVER=your_production_mssql_server
MSSQL_DATABASE=your_production_mssql_database
MSSQL_USERNAME=your_production_mssql_username
MSSQL_PASSWORD=your_production_mssql_password

# ===========================================
# REDIS CACHING (OPTIONAL)
# ===========================================
REDIS_CLOUD=true
REDIS_HOST=your_production_redis_host
REDIS_PORT=6379
REDIS_USER=default
REDIS_PASSWORD=your_production_redis_password
REDIS_DB=0
CACHE_TTL=10

# ===========================================
# SERVER CONFIGURATION
# ===========================================
ENVIRONMENT=production
DEV_BACKEND_URL=https://localhost:8200
PROD_BACKEND_URL=https://your-production-domain.com:8200

# Authentication (CHANGE THESE IN PRODUCTION!)
SECRET_KEY=your_secure_secret_key_change_this_in_production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Voice Agent Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8002
LOG_LEVEL=INFO
```

## SSL Configuration

For HTTPS in production:

1. Obtain SSL certificates (Let's Encrypt, commercial CA)
2. Update backend configuration:
   ```env
   SSL_CERT_FILE=/path/to/certificate.crt
   SSL_KEY_FILE=/path/to/private.key
   ```
3. Configure reverse proxy (Nginx, Apache) with SSL termination

## Monitoring and Logging

### Backend Logging

The application logs to stdout/stderr by default. For production:

1. Use a log management service (Datadog, Loggly, etc.)
2. Configure structured logging in the application
3. Set up log rotation

### Health Checks

The application provides health check endpoints:

- Backend: `GET /health` (when implemented)
- Database connectivity check
- AI service connectivity check

### Performance Monitoring

1. Set up application performance monitoring (APM)
2. Monitor database query performance
3. Track AI API usage and costs

## Backup and Recovery

### Database Backup

1. Set up regular database backups
2. Test backup restoration procedures
3. Store backups in secure, geographically distributed locations

### Configuration Backup

1. Version control all configuration files
2. Store secrets securely (HashiCorp Vault, AWS Secrets Manager)
3. Document recovery procedures

## Scaling

### Horizontal Scaling

1. **Backend**: Scale stateless API instances
2. **Frontend**: Scale web server instances
3. **Database**: Use read replicas for read-heavy workloads

### Vertical Scaling

1. Increase instance sizes for CPU/memory
2. Optimize database indexes and queries
3. Implement caching strategies

## Security Considerations

1. **Network Security**:
   - Use firewalls to restrict access
   - Implement network segmentation
   - Use private networks where possible

2. **Application Security**:
   - Keep dependencies updated
   - Implement proper input validation
   - Use parameterized queries to prevent SQL injection

3. **Data Security**:
   - Encrypt data at rest and in transit
   - Implement proper access controls
   - Regularly audit permissions

4. **API Security**:
   - Implement rate limiting
   - Use proper authentication and authorization
   - Validate all API inputs

## Troubleshooting

### Common Issues

1. **Database Connection Failed**:
   - Check database credentials
   - Verify network connectivity
   - Ensure database is running

2. **AI API Errors**:
   - Check API keys
   - Verify quota limits
   - Check network connectivity to API endpoints

3. **Performance Issues**:
   - Monitor resource usage
   - Check database query performance
   - Implement caching where appropriate

### Logs and Diagnostics

1. Check Docker container logs:
   ```bash
   docker-compose logs backend
   docker-compose logs frontend
   ```

2. Check application logs in `/var/log/easy-query/` (if configured)

3. Monitor system resources (CPU, memory, disk I/O)

## Maintenance

### Regular Tasks

1. Update dependencies regularly
2. Rotate secrets and API keys
3. Review and update security configurations
4. Monitor and optimize database performance
5. Review logs for errors and anomalies

### Updates

1. Backup current deployment
2. Test updates in staging environment
3. Deploy updates during maintenance windows
4. Monitor application after updates