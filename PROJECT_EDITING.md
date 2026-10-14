# 项目介绍与效果展示编辑说明

这份说明用于丰富董沁雨个人网站中的项目介绍，重点补充框架图、实验效果、演示动画，以及论文、代码或项目链接。

## 1. 找到自己的项目

首页只展示简短介绍，鼠标悬停或手机点按项目后才显示新窗口链接选项。图表、视频及详细说明请编辑独立项目页，不直接加入首页。

用 VS Code 等文本编辑器打开下表对应的详情页，编辑其中 `data-project="项目名"` 的 `<article>`。首页区块仍可通过 `PROJECT 项目名: START` 注释定位，用于调整简短介绍或链接选项。

| 项目 | 独立详情页 | 首页定位 ID | 素材文件夹 |
| --- | --- | --- | --- |
| WorldPrior 世界模型 | 尚未建立，先回传素材及说明 | `project-worldprior` | `assets/projects/worldprior/` |
| AutoPlanax 自动化 RL 研究 | `projects/autoplanax.html` | `project-autoplanax` | `assets/projects/autoplanax/` |
| Planax 并行仿真平台 | `projects/planax.html` | `project-planax` | `assets/projects/planax/` |
| G-EATS 固定翼控制 | `projects/g-eats.html` | `project-g-eats` | `assets/projects/g-eats/` |
| 数据采集与仿真回放 | `projects/replay.html` | `project-replay` | `assets/projects/replay/` |
| 四轴飞轮姿态控制 | 尚未建立，先回传素材及说明 | `project-reaction-wheels` | `assets/projects/reaction-wheels/` |

请保留详情页内的 `overview`、`framework`、`results`、`demos`（有视频时）和 `data` 定位 ID，首页选项会链接到这些位置。保留首页项目 ID 和区块起止注释，便于统一合并。

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

详情页位于 `projects/`，素材地址需返回上一层，例如 `../assets/projects/planax/framework.webp`。首页的链接地址以 `projects/planax.html?lang=zh#framework` 为例，英文对应 `?lang=en`，并设置 `target="_blank"`。

## 4. 加图片或 GIF

在独立详情页中找到已有的媒体容器；新增详情页时可使用：

```html
<div class="project-media" data-project-media></div>
```

把下面的 `<figure>` 放到容器里面。示例以 Planax 为例，文件名需要替换为自己实际提供的素材。

```html
<div class="project-media" data-project-media>
  <figure class="project-media-item is-wide">
    <img src="../assets/projects/planax/framework.webp"
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

图片内含中英文文字时，可以提供两份图片：`src` 指向中文版，`data-en-src` 指向英文版。点击打开的大图链接同样用 `href` 和 `data-en-href` 分别指定，切换语言时会一起更新。例如：

```html
<a href="../assets/projects/planax/framework-zh.svg"
   data-en-href="../assets/projects/planax/framework-en.svg"
   target="_blank" rel="noopener noreferrer">
  <img src="../assets/projects/planax/framework-zh.svg"
       data-en-src="../assets/projects/planax/framework-en.svg"
       alt="Planax 框架图" data-en-alt="Planax architecture"
       loading="lazy">
</a>
```

一张图会占满宽度；多张图默认在桌面端排列为两列、手机端排列为一列。添加 `is-wide` 可以让框架图或宽屏演示占满整行。

## 5. 加 MP4 视频

同样放到独立详情页的 `.project-media` 容器里。下面的版本有播放控件，访客点击后播放：

```html
<figure class="project-media-item is-wide">
  <video controls playsinline preload="metadata"
         poster="../assets/projects/planax/swarm-poster.webp"
         aria-label="无人机集群仿真演示"
         data-en-aria-label="UAV swarm simulation demonstration">
    <source src="../assets/projects/planax/swarm-demo.mp4" type="video/mp4">
  </video>
  <figcaption>
    <span data-en="A demonstration of UAV swarm interaction in simulation.">无人机集群交互的仿真演示。</span>
  </figcaption>
</figure>
```

如需静音自动循环展示，可以把 `<video>` 的属性改为 `autoplay muted loop playsinline controls preload="metadata"`。网站会在访客启用“减少动态效果”时关闭视频自动播放。没有封面图时，删除 `poster` 属性。

## 6. 加论文、代码或演示链接

在详情页的“数据与相关链接”中添加或更新链接。首页新增选项时，放入对应项目的 `[data-project-options]` 容器内，使其只在触碰项目后显示。下面使用已有的 Planax 代码地址示范：

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

1. 保存文件，双击 `index.html` 打开浏览器，悬停或点按自己的项目，确认链接选项在触碰后显示。
2. 点击右上角 `EN`，检查英文文字与排版。
3. 缩窄浏览器窗口，检查图片、视频和标题是否适合手机阅读。
4. 点击选项，确认在新窗口打开对应详情页，播放视频并核对文件路径及图片显示。
5. 将修改内容打包回传给董沁雨。

每位同学的回传包建议包含：

```text
planax-修改回传/
  projects/planax.html                # 自己项目的完整详情页
  assets/projects/planax/             # 自己项目新增或修改的素材
  修改说明.txt                       # 变更内容、素材顺序、链接与实验说明
```

如修改了首页链接选项，一并回传 `index.html` 并注明项目区块。如额外修改了 `styles.css` 或 `script.js`，一并回传并注明变更位置。可再附一张预览截图，方便核对。

各同学回传的详情页和素材会由董沁雨统一合并后发布。

不编辑 HTML 时，也可以只回传素材文件夹和一份说明，列明：每个文件的中英文图注、展示顺序、论文/代码/视频链接，以及希望补充的中英文介绍。
