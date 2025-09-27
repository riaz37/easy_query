# Easy Query

<div align="center">
  <img src="./public/logo/logo.svg" alt="Easy Query Logo" width="120" height="120">
  
  **A powerful AI-driven knowledge base and database query platform**
  
  [![Next.js](https://img.shields.io/badge/Next.js-15.3.3-black?logo=next.js)](https://nextjs.org/)
  [![React](https://img.shields.io/badge/React-19.0.0-blue?logo=react)](https://reactjs.org/)
  [![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?logo=typescript)](https://www.typescriptlang.org/)
  [![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4.0-38B2AC?logo=tailwind-css)](https://tailwindcss.com/)
</div>

---

## 🚀 Overview

Easy Query is a sophisticated AI-powered platform that revolutionizes how users interact with databases and documents. It combines natural language processing, voice commands, and intelligent query generation to make data analysis accessible to everyone.

### ✨ Key Features

- **🧠 AI-Powered Querying** - Convert natural language to SQL queries automatically
- **🎤 Voice Navigation** - Control the entire application using voice commands
- **📄 Document Intelligence** - Upload and query documents (PDF, Excel, Word, etc.)
- **📊 Advanced Reporting** - Generate comprehensive reports with visualizations
- **🏢 Multi-Company Support** - Hierarchical company structure management
- **👥 User Management** - Role-based access control and user configuration
- **📈 Real-time Analytics** - Live query execution with progress tracking
- **🎨 Modern UI/UX** - Beautiful glassmorphism design with smooth animations
- **🌙 Dark/Light Theme** - Customizable theme with smooth transitions
- **📱 Responsive Design** - Works seamlessly across all devices

---

## 🏗️ Architecture

### Core Technologies

- **Frontend**: Next.js 15, React 19, TypeScript
- **Styling**: Tailwind CSS v4, ShadCN UI Components
- **State Management**: Zustand
- **Animations**: Framer Motion, Three.js
- **Charts**: Recharts
- **Voice Processing**: Pipecat AI
- **PDF Generation**: jsPDF
- **File Processing**: XLSX, React Dropzone

### Key Components

```
src/
├── app/                    # Next.js App Router pages
├── components/            # React components
│   ├── ai-reports/       # AI report generation
│   ├── auth/             # Authentication components
│   ├── database-query/   # Database querying interface
│   ├── file-query/       # Document querying interface
│   ├── tables/           # Table management
│   ├── ui/               # Reusable UI components
│   └── voice-agent/      # Voice command handling
├── lib/                   # Utility functions and services
│   ├── api/              # API service layer
│   ├── hooks/            # Custom React hooks
│   └── voice-agent/      # Voice processing logic
├── store/                 # Global state management
└── types/                 # TypeScript type definitions
```

---

## 🚦 Getting Started

### Prerequisites

- **Node.js** 18.0 or higher
- **npm** or **pnpm** package manager
- Modern web browser with JavaScript enabled

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/easy-query.git
   cd easy-query
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   pnpm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env.local
   ```
   
   Configure your environment variables:
   ```env
   # API Configuration
   NEXT_PUBLIC_API_BASE_URL=your_api_url
   
   # Database Configuration
   DATABASE_URL=your_database_url
   
   # Authentication
   NEXTAUTH_SECRET=your_secret_key
   NEXTAUTH_URL=http://localhost:3000
   
   # Voice Agent (Optional)
   PIPECAT_API_KEY=your_pipecat_key
   ```

4. **Run the development server**
   ```bash
   npm run dev
   # or
   pnpm dev
   ```

5. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

---

## 📖 Usage Guide

### 🗄️ Database Querying

1. **Connect to Database**
   - Navigate to the Dashboard
   - Configure your database connection
   - Select your preferred database

2. **Natural Language Queries**
   ```
   "Show me all customers from last month"
   "What are the top 5 selling products?"
   "Find users with email containing 'gmail'"
   ```

3. **Advanced Features**
   - Query history and favorites
   - Export results to Excel/PDF
   - Collaborative query sharing

### 📄 Document Intelligence

1. **Upload Documents**
   - Drag & drop files or click to browse
   - Supported formats: PDF, DOCX, XLSX, CSV, TXT

2. **Query Documents**
   ```
   "Summarize this document"
   "Extract all dates and amounts"
   "What are the key recommendations?"
   ```

3. **Advanced Analysis**
   - Cross-document queries
   - Data extraction and visualization
   - Report generation

### 🎤 Voice Commands

Activate voice mode and use commands like:

- **Navigation**: "Go to dashboard", "Open database query"
- **Actions**: "Upload a file", "Generate report"
- **Queries**: "Show me sales data", "Find customer information"

---

## 🛠️ Development

### Project Structure

```
easy-query/
├── 📁 src/
│   ├── 🎨 app/                 # Next.js pages and layouts
│   ├── 🧩 components/          # Reusable React components
│   ├── 📚 lib/                 # Utilities and services
│   ├── 🗃️ store/               # State management
│   └── 📝 types/               # TypeScript definitions
├── 📁 public/                  # Static assets
├── 📄 components.json          # ShadCN UI configuration
├── 📄 tailwind.config.js       # Tailwind CSS configuration
└── 📄 tsconfig.json            # TypeScript configuration
```

### Available Scripts

```bash
# Development
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint

# Type Checking
npm run type-check   # Run TypeScript compiler
```

### Code Style

- **ESLint** for code linting
- **Prettier** for code formatting
- **TypeScript** for type safety
- **Tailwind CSS** for styling

---

## 🎨 Design System

### Theme Configuration

The application features a modern design system with:

- **Colors**: Professional emerald green theme with dark mode support
- **Typography**: Barlow for headings, Public Sans for body text
- **Components**: Consistent design patterns using ShadCN UI
- **Animations**: Smooth transitions and micro-interactions

### Customization

Modify the theme in:
- `src/app/globals.css` - CSS custom properties
- `components.json` - ShadCN UI configuration
- `src/store/theme-store.ts` - Theme state management

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NEXT_PUBLIC_API_BASE_URL` | Backend API URL | Yes |
| `DATABASE_URL` | Database connection string | Yes |
| `NEXTAUTH_SECRET` | Authentication secret key | Yes |
| `PIPECAT_API_KEY` | Voice processing API key | No |

### Database Setup

1. Configure your database connection
2. Run migrations (if applicable)
3. Set up user authentication
4. Configure business rules and permissions

---

## 🚀 Deployment

### Production Build

```bash
npm run build
npm run start
```

### Platform Deployment

#### Vercel (Recommended)
```bash
npm install -g vercel
vercel --prod
```

#### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

#### Environment Setup
Ensure all environment variables are configured in your deployment platform.

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Process

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Run tests and linting**
   ```bash
   npm run lint
   npm run type-check
   ```
5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to your branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **ShadCN UI** for the beautiful component library
- **Tailwind CSS** for the utility-first CSS framework
- **Next.js Team** for the amazing React framework
- **Pipecat AI** for voice processing capabilities
- **All contributors** who have helped make this project better

---

## 📞 Support

- **Documentation**: [docs.easyquery.com](https://docs.easyquery.com)
- **Issues**: [GitHub Issues](https://github.com/your-username/easy-query/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/easy-query/discussions)
- **Email**: support@easyquery.com

---

<div align="center">
  <p>Made with ❤️ by the Easy Query Team</p>
  <p>
    <a href="https://github.com/your-username/easy-query">⭐ Star us on GitHub</a> •
    <a href="https://twitter.com/easyquery">🐦 Follow on Twitter</a> •
    <a href="https://linkedin.com/company/easyquery">💼 LinkedIn</a>
  </p>
</div>
