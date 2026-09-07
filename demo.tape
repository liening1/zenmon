# VHS Tape for ZenMon GitHub README Demo
# Run: vhs demo.tape

Output demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 720
Set Padding 15
Set Theme "Catppuccin Mocha"

# 1. 启动 zenmon
Type "zenmon"
Sleep 500ms
Enter
Sleep 3.5s

# 2. 演示宽屏切换 (w)
Type "w"
Sleep 2.5s
Type "w"
Sleep 2.0s

# 3. 演示 CPU 矩阵 (1)
Type "1"
Sleep 2.5s

# 4. 演示 GPU 深度遥测 (2)
Type "2"
Sleep 2.5s

# 5. 演示 历史波形 (5)
Type "5"
Sleep 3.0s

# 6. 切回主面板并退出
Type "0"
Sleep 2.0s
Type "q"
Sleep 500ms
