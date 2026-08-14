# cosdt-ci-test/workflows

Example CI 的唯一部署点，红绿只出现在本仓的 Actions。每个被看护的项目在根目录有一个同名目录存放配套清单、脚本与数据；流水线本身放在 `.github/workflows/<项目>.yml`。

```
.github/workflows/ms-swift.yml   ms-swift 流水线
ms-swift/                        ms-swift 的清单、overlay、fixture、脚本
```
