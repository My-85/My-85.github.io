# 董沁雨个人学术网站

线上地址：https://my-85.github.io/

这是一个直接使用 HTML、CSS 和 JavaScript 的静态网站。首次访问默认中文，右上角“中文 / EN”可以切换语言，并记住访客的选择。

## 如何预览

解压源码包后，双击 `index.html` 即可查看页面和切换语言，不需要安装 Node.js 或构建工具。

如果希望用本地服务预览，在源码文件夹打开终端，运行：

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

浏览器打开 `http://127.0.0.1:8000/`。在终端按 `Ctrl+C` 停止服务。

可以使用这些地址检查指定语言和项目：

- 中文：`http://127.0.0.1:8000/?lang=zh#projects`
- 英文：`http://127.0.0.1:8000/?lang=en#projects`
- 单个项目：`http://127.0.0.1:8000/?lang=zh#project-worldprior`

## 文件用途

| 文件或文件夹 | 用途 |
| --- | --- |
| `index.html` | 页面正文、项目介绍、中英文内容和媒体展示位置 |
| `styles.css` | 排版、手机适配、项目图片与视频样式 |
| `script.js` | 语言切换、语言记忆、手机菜单和滚动显示 |
| `assets/` | 页面使用的照片、图标、PDF 等资源 |
| `assets/projects/项目名/` | 各项目新增的图片、GIF、MP4 等素材 |
| `PROJECT_EDITING.md` | 项目同学的具体编辑步骤及可复制模板 |
| `HANDOFF_MESSAGE.txt` | 发给项目同学时可以直接复制的说明 |
| `.nojekyll` | GitHub Pages 静态部署配置 |

## 项目同学从哪里开始

请先看 `PROJECT_EDITING.md`，然后在 `index.html` 中搜索自己项目的 `PROJECT ...: START` 注释。

每位同学编辑自己项目的完整 `<article>` 区块，并把素材放到对应项目文件夹。修改完成后，将编辑过的文件、素材和修改说明回传给董沁雨，由董沁雨统一合并与发布。

不熟悉 HTML 的同学也可以直接提供素材、中文及英文说明、链接和展示顺序，供统一整合。

## 中英文内容怎样写

中文直接写在标签里面，英文写在同一个标签的 `data-en` 属性中：

```html
<span data-en="Closed-loop control results">闭环控制效果</span>
```

图片替代文字同样支持双语：

```html
<img src="assets/projects/planax/result.webp"
     alt="无人机集群仿真效果"
     data-en-alt="UAV swarm simulation results">
```

`data-en` 用于纯文字。如果一段文字含有 `<strong>`、链接等子标签，请分别为子标签中的文字添加 `data-en`，保持原有结构。

## 发布

网站仓库：https://github.com/My-85/My-85.github.io

由董沁雨将核对后的修改提交并推送到仓库 `main` 分支，GitHub Pages 会自动发布。项目同学在本地预览并回传修改即可。
