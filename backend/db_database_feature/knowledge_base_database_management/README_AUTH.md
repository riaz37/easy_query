# Authentication and Sign Out Feature

This document describes the authentication system and sign out functionality implemented in the Knowledge Base Database Management API.

## Features

### Authentication System
- **User Registration**: Create new user accounts with username, email, and password
- **User Login**: Authenticate users and receive JWT access tokens
- **Token-based Authentication**: Secure API endpoints using JWT tokens
- **Token Revocation**: Sign out functionality that invalidates JWT tokens

### Sign Out Features
- **Single Session Logout**: Revoke the current user's token
- **Token Blacklisting**: Store revoked tokens in database to prevent reuse
- **Automatic Token Validation**: Check for revoked tokens on each authenticated request

## API Endpoints

### Authentication Endpoints

#### POST `/signup`
Register a new user account.

**Request Body:**
```json
{
    "username": "string",
    "email": "user@example.com",
    "password": "string"
}
```

**Response:**
```json
{
    "id": 1,
    "username": "string",
    "email": "user@example.com",
    "created_at": "2024-01-01T00:00:00"
}
```

#### POST `/login`
Authenticate user and receive access token.

**Request Body:**
```json
{
    "username": "string",
    "password": "string"
}
```

**Response:**
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer"
}
```

### Sign Out Endpoints

#### POST `/logout`
Sign out the current user by revoking their token.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
    "message": "Successfully signed out",
    "success": true
}
```

#### POST `/logout-all`
Sign out from all sessions (current implementation revokes current token).

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
    "message": "Successfully signed out from all sessions",
    "success": true
}
```

### Protected Endpoints

#### GET `/profile`
Get current user profile information.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
    "id": 1,
    "username": "string",
    "email": "user@example.com",
    "created_at": "2024-01-01T00:00:00"
}
```

#### GET `/protected`
Example protected route that requires authentication.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
    "message": "Hello username, this is a protected route!"
}
```

## Database Schema

### Users Table
```sql
CREATE TABLE dbo.users (
    id INT PRIMARY KEY IDENTITY(1,1),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at DATETIME2 DEFAULT GETDATE()
);
```

### Revoked Tokens Table
```sql
CREATE TABLE dbo.revoked_tokens (
    id INT PRIMARY KEY IDENTITY(1,1),
    jti VARCHAR(255) UNIQUE NOT NULL,
    revoked_at DATETIME2 DEFAULT GETDATE()
);
```

## Security Features

### JWT Token Structure
- **JTI (JWT ID)**: Unique identifier for each token
- **Expiration**: Tokens expire after 30 minutes
- **Algorithm**: HS256 for token signing
- **Secret Key**: Secure secret key for token validation

### Token Revocation Process
1. User requests logout with valid token
2. System extracts JTI from token
3. JTI is stored in `revoked_tokens` table
4. Subsequent requests with same token are rejected

### Authentication Flow
1. User logs in and receives JWT token
2. Token is included in Authorization header for protected requests
3. System validates token and checks if it's revoked
4. If valid and not revoked, request proceeds
5. If invalid or revoked, request is rejected with 401 error

## Usage Examples

### Using the API with curl

#### 1. Register a new user
```bash
curl -X POST "http://localhost:8000/signup" \
     -H "Content-Type: application/json" \
     -d '{"username": "testuser", "email": "test@example.com", "password": "password123"}'
```

#### 2. Login and get token
```bash
curl -X POST "http://localhost:8000/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "testuser", "password": "password123"}'
```

#### 3. Access protected route
```bash
curl -X GET "http://localhost:8000/protected" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

#### 4. Sign out
```bash
curl -X POST "http://localhost:8000/logout" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Using the test script
```bash
python test_auth.py
```

## Configuration

### Environment Variables
- `SECRET_KEY`: JWT secret key (default: "esap-secret-key-kb-esap-secret-key-23334-KB")
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time in minutes (default: 30)

### Database Configuration
The authentication system uses the same database configuration as the main application. Ensure the database connection is properly configured in `database.py`.

## Error Handling

### Common Error Responses

#### 401 Unauthorized
- Invalid or expired token
- Token has been revoked
- Missing Authorization header

#### 400 Bad Request
- Invalid request body
- Missing required fields
- Invalid token format

#### 404 Not Found
- User not found
- Resource not found

#### 500 Internal Server Error
- Database connection issues
- Server configuration problems

## Best Practices

1. **Token Storage**: Store tokens securely on the client side
2. **Token Expiration**: Implement token refresh mechanism for long-running applications
3. **HTTPS**: Always use HTTPS in production
4. **Password Security**: Use strong passwords and consider implementing password policies
5. **Rate Limiting**: Implement rate limiting for authentication endpoints
6. **Logging**: Log authentication events for security monitoring

## Future Enhancements

1. **Refresh Tokens**: Implement refresh token mechanism
2. **Session Management**: Track multiple user sessions
3. **Password Reset**: Add password reset functionality
4. **Two-Factor Authentication**: Implement 2FA support
5. **Role-Based Access Control**: Add user roles and permissions
6. **Audit Logging**: Enhanced logging for security events
