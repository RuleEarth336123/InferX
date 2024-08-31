<div id="top"></div>
<div align="center">
 <img alt="inferx" height="200px" src="res/ico.png">
</div>
<!-- 项目 LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
   
  </a>

  <h3 align="center">InferX</h3>

  <p align="center">
    异构平台大模型推理框架
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>浏览文档 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">查看 Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">反馈 Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">请求新功能</a>
  </p>
</div>



<!-- 目录 -->
<details>
  <summary>目录</summary>
  <ol>
    <li>
      <a href="#关于本项目">关于本项目</a>
      <ul>
        <li><a href="#构建工具">构建工具</a></li>
      </ul>
    </li>
    <li>
      <a href="#开始">开始</a>
      <ul>
        <li><a href="#依赖">依赖</a></li>
        <li><a href="#安装">安装</a></li>
      </ul>
    </li>
    <li><a href="#使用方法">使用方法</a></li>
    <li><a href="#路线图">路线图</a></li>
    <li><a href="#贡献">贡献</a></li>
    <li><a href="#许可证">许可证</a></li>
    <li><a href="#联系我们">联系我们</a></li>
    <li><a href="#致谢">致谢</a></li>
  </ol>
</details>



<!-- 关于本项目 -->
## 关于本项目

[![产品截图][product-screenshot]](https://example.com)

大语言模型推理框架的异构实现，可在amd架构和nvidia gpu的加速推理,支持windows、linux以及android跨平台编译，可满足大模型端侧推理的需求

<p align="right">(<a href="#top">返回顶部</a>)</p>

### 文件目录说明
eg:

```
llama-infer 
├── LICENSE.txt
├── README.md
├── /include/
│  ├── /base/
│  ├── /op/
│  ├── /tensor/
│  ├── /models/
│  ├── /samples/
│  ├── /helper/
├── /src/
│  ├── /base/
│  ├── ...
├── /kernels/
│  ├── /cpu/
│  ├── /cuda/
│  ├── kernel_interface
├── /docs/
│  ├── /rules/
│  │  ├── backend.txt
│  │  └── frontend.txt
├── /scripts/
├── /lib/
├── /thirds/
├── /log/
└── /util/

```

### 构建工具

cmake make gcc g++ cuda

* [Next.js](https://nextjs.org/)
* [React.js](https://reactjs.org/)
* [Vue.js](https://vuejs.org/)
* [Angular](https://angular.io/)
* [Svelte](https://svelte.dev/)
* [Laravel](https://laravel.com)
* [Bootstrap](https://getbootstrap.com)
* [JQuery](https://jquery.com)

<p align="right">(<a href="#top">返回顶部</a>)</p>



<!-- 开始 -->
## 开始



### 依赖

* sh
  ```sh
  sh ./scripts/install.sh
  ```
* apt
  ```sh
  sh ./scripts/install.sh
  ```


### 安装

_下面是一个指导你的受众如何安装和配置你的应用的例子。这个模板不需要任何外部依赖或服务。_

1. 在 [https://example.com](https://example.com) 获取一个免费的 API Key。
2. 克隆本仓库
   ```sh
   git clone -b v1.0 https://gitee.com/concenmo/llama-infer
   ```
3. 安装 NPM 包
   ```sh
   npm install
   ```
4. 编译该项目
    ```sh
    mkdir build
    cd build
    cmake --DCMAKE_BUILD_TYPE=Release ..
    make -j10
    ```

<p align="right">(<a href="#top">返回顶部</a>)</p>



<!-- 使用方法 示例 -->
## 使用方法

1.  下载模型，地址：https://hf-mirror.com/models
2.  启动命令：./main models 14132


_转到 [文档](https://example.com) 查看更多示例_

<p align="right">(<a href="#top">返回顶部</a>)</p>

## 客户端使用

### restful api

```
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.1",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}'
```

### openai

```
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Say this is a test',
        }
    ],
    model='llama3',
)

response = client.chat.completions.create(
    model="llava",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": "iVBORw0KGgoAAAANSUhEUgAAAG0AAABmCAYAAADBPx
                }
            ],
        }
    ],
    max_tokens=300,
)

completion = client.completions.create(
    model="llama3",
    prompt="Say this is a test",
)

list_completion = client.models.list()

model = client.models.retrieve("llama3")

embeddings = client.embeddings.create(
    model="all-minilm",
    input=["why is the sky blue?", "why is the grass green?"],
)
```




<!-- 路线图 -->
## 路线图

- [x] 添加更新日志
- [x] 添加「返回顶部」链接
- [ ] 添加额外的模板和示例
- [ ] 添加「组件」文档，以便更容易复制和粘贴各个部分
- [ ] 多语种支持
    - [x] 中文
    - [ ] 西班牙语

到 [open issues](https://github.com/othneildrew/Best-README-Template/issues) 页查看所有请求的功能 （以及已知的问题）。

<p align="right">(<a href="#top">返回顶部</a>)</p>



<!-- 贡献 -->
## 贡献

贡献让开源社区成为了一个非常适合学习、启发和创新的地方。你所做出的任何贡献都是**受人尊敬**的。

如果你有好的建议，请复刻（fork）本仓库并且创建一个拉取请求（pull request）。你也可以简单地创建一个议题（issue），并且添加标签「enhancement」。不要忘记给项目点一个 star！再次感谢！

1. 复刻（Fork）本项目
2. 创建你的 Feature 分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的变更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到该分支 (`git push origin feature/AmazingFeature`)
5. 创建一个拉取请求（Pull Request）

<p align="right">(<a href="#top">返回顶部</a>)</p>



<!-- 许可证 -->
## 许可证

根据 MIT 许可证分发。打开 [LICENSE.txt](LICENSE.txt) 查看更多内容。


<p align="right">(<a href="#top">返回顶部</a>)</p>



<!-- 联系我们 -->
## 联系我们

你的名字 - [@your_twitter](https://twitter.com/your_username) - email@example.com

项目链接: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#top">返回顶部</a>)</p>



<!-- 致谢 -->
## 致谢

在这里列出你觉得有用的资源，并以此致谢。我已经添加了一些我喜欢的资源，以便你可以快速开始！

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#top">返回顶部</a>)</p>



<!-- MARKDOWN 链接 & 图片 -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/BreakingAwful/Best-README-Template-zh.svg?style=for-the-badge
[contributors-url]: https://github.com/BreakingAwful/Best-README-Template-zh/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/BreakingAwful/Best-README-Template-zh.svg?style=for-the-badge
[forks-url]: https://github.com/BreakingAwful/Best-README-Template-zh/network/members
[stars-shield]: https://img.shields.io/github/stars/BreakingAwful/Best-README-Template-zh.svg?style=for-the-badge
[stars-url]: https://github.com/BreakingAwful/Best-README-Template-zh/stargazers
[issues-shield]: https://img.shields.io/github/issues/BreakingAwful/Best-README-Template-zh.svg?style=for-the-badge
[issues-url]: https://github.com/BreakingAwful/Best-README-Template-zh/issues
[license-shield]: https://img.shields.io/github/license/BreakingAwful/Best-README-Template-zh.svg?style=for-the-badge
[license-url]: https://github.com/BreakingAwful/Best-README-Template-zh/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
