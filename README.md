# 董沁雨个人学术网站

线上地址：https://my-85.github.io/

这是一个直接使用 HTML、CSS 和 JavaScript 的静态网站。首次访问默认中文，右上角“中文 / EN”可以切换语言，并记住访客的选择。

首页保留简洁的项目介绍。鼠标悬停或手机点按相应项目后，才显示资料链接；选择链接会在新窗口打开独立项目页。框架图、结果图、视频和实验数据放在独立项目页中。

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
- 独立项目页：`http://127.0.0.1:8000/projects/planax.html?lang=zh`

## 文件用途

| 文件或文件夹 | 用途 |
| --- | --- |
| `index.html` | 首页正文、简短项目介绍及悬停／点按后显示的资料链接 |
| `projects/项目名.html` | 独立项目页，包含详细介绍、框架图、结果图和视频 |
| `styles.css` | 排版、手机适配、项目图片与视频样式 |
| `script.js` | 语言切换、语言记忆、项目资料选项、手机菜单和滚动显示 |
| `assets/` | 页面使用的照片、图标、PDF 等资源 |
| `assets/projects/项目名/` | 各项目新增的图片、GIF、MP4 等素材 |
| `PROJECT_EDITING.md` | 项目同学的具体编辑步骤及可复制模板 |
| `HANDOFF_MESSAGE.txt` | 发给项目同学时可以直接复制的说明 |
| `.nojekyll` | GitHub Pages 静态部署配置 |

## 项目同学从哪里开始

请先看 `PROJECT_EDITING.md`，然后打开自己的独立项目页：`projects/autoplanax.html`、`projects/planax.html`、`projects/g-eats.html` 或 `projects/replay.html`。

每位同学编辑自己的项目详情页，并把素材放到对应项目文件夹。首页只保留简短介绍和隐藏的链接选项。WorldPrior 和四轴飞轮项目尚未建立详情页，可先回传素材及说明。修改完成后，将编辑过的文件、素材和修改说明回传给董沁雨，由董沁雨统一合并与发布。

不熟悉 HTML 的同学也可以直接提供素材、中文及英文说明、链接和展示顺序，供统一整合。

## 中英文内容怎样写

中文直接写在标签里面，英文写在同一个标签的 `data-en` 属性中：

```html
<span data-en="Closed-loop control results">闭环控制效果</span>
```

图片替代文字同样支持双语：

```html
<img src="../assets/projects/planax/result.webp"
     alt="无人机集群仿真效果"
     data-en-alt="UAV swarm simulation results">
```

`data-en` 用于纯文字。如果一段文字含有 `<strong>`、链接等子标签，请分别为子标签中的文字添加 `data-en`，保持原有结构。

图中含有中英文文字时，用 `src` / `data-en-src` 指定两种语言的图片，用 `href` / `data-en-href` 指定对应的大图链接。具体示例见 `PROJECT_EDITING.md`。

## 发布

网站仓库：https://github.com/My-85/My-85.github.io

由董沁雨将核对后的修改提交并推送到仓库 `main` 分支，GitHub Pages 会自动发布。项目同学在本地预览并回传修改即可。
