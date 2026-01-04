# Eidos 归因系统前端

基于 React + TypeScript + Tailwind CSS + Vite 构建的回测归因系统前端应用。

## 技术栈

- **React 18** - UI 框架
- **TypeScript** - 类型安全
- **Tailwind CSS** - 样式框架
- **Vite** - 构建工具
- **React Router** - 路由管理
- **Axios** - HTTP 客户端
- **Recharts** - 图表库
- **date-fns** - 日期处理

## 开发

### 安装依赖

```bash
npm install
```

### 启动开发服务器

```bash
npm run dev
```

应用将在 `http://localhost:3000` 启动。

### 构建生产版本

```bash
npm run build
```

构建产物将输出到 `dist/` 目录。

### 预览生产构建

```bash
npm run preview
```

## 项目结构

```
web/eidos/
├── src/
│   ├── components/      # 可复用组件
│   ├── pages/           # 页面组件
│   ├── services/        # API 服务
│   ├── types/           # TypeScript 类型定义
│   ├── utils/           # 工具函数
│   ├── App.tsx          # 主应用组件
│   └── main.tsx         # 入口文件
├── public/              # 静态资源
└── dist/                # 构建输出（gitignore）
```

## API 配置

前端通过 Vite 代理连接到后端 API。默认配置：

- 开发环境：`http://localhost:8000/api/v1`
- 生产环境：需要配置环境变量或修改 `vite.config.ts` 中的 `baseURL`

## 环境变量

创建 `.env` 文件配置环境变量：

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

## 代码规范

项目使用 ESLint 进行代码检查：

```bash
npm run lint
```

