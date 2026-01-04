/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        eidos: {
          bg: '#0B0E14',      // 深海黑 (Obsidian) - 整个系统的底层背景
          surface: '#161B22', // 钛灰色 (Titanium) - 卡片、容器、侧边栏
          gold: '#C5A059',    // 雅典金 (Eidos Gold) - Logo、核心标题、激活状态
          accent: '#00F2FF',  // 电光青 (Cyan) - 正向贡献、净值上涨、超配
          danger: '#FF3D00',  // 胭脂红 (Carmine) - 负向贡献、净值下跌、低配
          muted: '#8B949E',   // 灰冷蓝 (Slate Blue) - 次要文字、坐标轴、非活跃标签
        },
      },
      backgroundImage: {
        'gold-gradient': 'linear-gradient(135deg, #C5A059 0%, #8E7037 100%)',
        'cyan-gradient': 'linear-gradient(135deg, #00F2FF 0%, #008CFF 100%)',
        'carmine-gradient': 'linear-gradient(135deg, #FF3D00 0%, #991B1B 100%)',
      },
      fontFamily: {
        'display': ['Cinzel', 'Playfair Display', 'serif'], // 主标题字体
        'mono': ['JetBrains Mono', 'Inter', 'monospace'],   // 数据/代码字体
      },
      backdropBlur: {
        'glass': '12px',
      },
    },
  },
  plugins: [],
}

