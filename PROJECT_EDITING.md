# 项目介绍与效果展示编辑说明

这份说明用于丰富董沁雨个人网站中的项目介绍，重点补充框架图、实验效果、演示动画，以及论文、代码或项目链接。

## 1. 找到自己的项目

用 VS Code 等文本编辑器打开 `index.html`，搜索下表中的区块起始注释。每个区块从 `START` 到 `END`，包含一个完整的 `<article>`。

| 项目 | 搜索关键词 | 页面定位 ID | 素材文件夹 |
| --- | --- | --- | --- |
| WorldPrior 世界模型 | `PROJECT worldprior: START` | `project-worldprior` | `assets/projects/worldprior/` |
| AutoPlanax 自动化 RL 研究 | `PROJECT autoplanax: START` | `project-autoplanax` | `assets/projects/autoplanax/` |
| Planax 并行仿真平台 | `PROJECT planax: START` | `project-planax` | `assets/projects/planax/` |
| G-EATS 固定翼控制 | `PROJECT g-eats: START` | `project-g-eats` | `assets/projects/g-eats/` |
| 数据采集与仿真回放 | `PROJECT replay: START` | `project-replay` | `assets/projects/replay/` |
| 四轴飞轮姿态控制 | `PROJECT reaction-wheels: START` | `project-reaction-wheels` | `assets/projects/reaction-wheels/` |

请在本项目区块中补充内容，保留项目 ID 和区块起止注释，便于统一合并各位同学的修改。

## 2. 推荐补充什么

- 一张框架图或系统结构图，说明方法、输入输出与各模块关系。
- 一段效果动画，展示控制、仿真、机器人任务或自动实验流程。
- 一张关键结果图或对比图，注明任务、基线、指标和实验条件。
- 论文、代码仓库、项目主页或演示视频的完整链接。
- 图文介绍围绕董沁雨在项目中承担的工作、方法与结果展开，提供中文和英文说明。

论文题目、投稿或录用状态、作者贡献和实验数字，请结合项目实际信息补充。

## 3. 放入素材

素材放到对应项目文件夹。文件名建议使用英文、数字和连字符，例如：

```text
assets/projects/planax/
  framework.webp
  swarm-demo.mp4
  swarm-poster.webp
  benchmark-results.png
```

图片可以使用 WebP、PNG、JPG 或 SVG；动画可以使用 GIF 或 MP4。展示型动画优先使用短 MP4，长视频放到视频平台并提供链接。图片建议宽度约 1200–1920 像素，文字和曲线保持清楚。

页面里的地址使用相对路径，例如 `assets/projects/planax/framework.webp`。

## 4. 加图片或 GIF

每个项目已提供一个空的媒体容器：

```html
<div class="project-media" data-project-media></div>
```

把下面的 `<figure>` 放到容器里面。示例以 Planax 为例，文件名需要替换为自己实际提供的素材。

```html
<div class="project-media" data-project-media>
  <figure class="project-media-item is-wide">
    <img src="assets/projects/planax/framework.webp"
         alt="Planax 仿真与训练框架"
         data-en-alt="Planax simulation and training architecture"
         loading="lazy">
    <figcaption>
      <span data-en="Overview of the parallel simulation and reinforcement-learning pipeline.">并行仿真与强化学习训练流程。</span>
    </figcaption>
  </figure>
</div>
```

GIF 的用法与图片相同，将 `src` 改为实际 `.gif` 文件路径即可。

一张图会占满宽度；多张图默认在桌面端排列为两列、手机端排列为一列。添加 `is-wide` 可以让框架图或宽屏演示占满整行。

## 5. 加 MP4 视频

同样放到项目的 `.project-media` 容器里。下面的版本有播放控件，访客点击后播放：

```html
<figure class="project-media-item is-wide">
  <video controls playsinline preload="metadata"
         poster="assets/projects/planax/swarm-poster.webp"
         aria-label="无人机集群仿真演示"
         data-en-aria-label="UAV swarm simulation demonstration">
    <source src="assets/projects/planax/swarm-demo.mp4" type="video/mp4">
  </video>
  <figcaption>
    <span data-en="A demonstration of UAV swarm interaction in simulation.">无人机集群交互的仿真演示。</span>
  </figcaption>
</figure>
```

如需静音自动循环展示，可以把 `<video>` 的属性改为 `autoplay muted loop playsinline controls preload="metadata"`。网站会在访客启用“减少动态效果”时关闭视频自动播放。没有封面图时，删除 `poster` 属性。

## 6. 加论文、代码或演示链接

在项目区块内添加或更新链接。下面使用已有的 Planax 代码地址示范：

```html
<p class="project-links">
  <a href="https://github.com/My-85/Aeroplanax"
     target="_blank" rel="noopener noreferrer">
    <span data-en="Code repository">代码仓库</span>
  </a>
</p>
```

更换 `href`，并将中文文字与 `data-en` 改为“论文 / Paper”“项目主页 / Project website”“演示视频 / Demo video”等。

## 7. 修改中英文文字

例如：

```html
<p>
  <span data-en="Describe the method and your contribution here.">在这里介绍方法与本人承担的工作。</span>
</p>
```

中文写在标签内部，英文写在 `data-en`。属性中的英文双引号写为 `&quot;`，`&` 写为 `&amp;`。新增图注和图片替代文字也应同时提供中英文。

## 8. 预览与回传

1. 保存文件，双击 `index.html` 打开浏览器，检查中文效果。
2. 点击右上角 `EN`，检查英文文字与排版。
3. 缩窄浏览器窗口，检查图片、视频和标题是否适合手机阅读。
4. 点击新增链接，播放视频，核对文件路径及图片是否显示。
5. 将修改内容打包回传给董沁雨。

每位同学的回传包建议包含：

```text
planax-修改回传/
  index.html                         # 保留完整页面，说明只修改了哪个项目区块
  assets/projects/planax/             # 自己项目新增或修改的素材
  修改说明.txt                       # 变更内容、素材顺序、链接与实验说明
```

如额外修改了 `styles.css` 或 `script.js`，一并回传并注明变更位置。可再附一张预览截图，方便核对。

各同学回传的项目区块会由董沁雨统一合并后发布。

不编辑 HTML 时，也可以只回传素材文件夹和一份说明，列明：每个文件的中英文图注、展示顺序、论文/代码/视频链接，以及希望补充的中英文介绍。
