"use strict";(self.webpackChunkedwardzjl_github_io=self.webpackChunkedwardzjl_github_io||[]).push([[1477],{10:e=>{e.exports=JSON.parse('{"blogPosts":[{"id":"the-downsides-of-json-for-config-files","metadata":{"permalink":"/blog/the-downsides-of-json-for-config-files","editUrl":"https://github.com/edwardzjl/edwardzjl.github.io/blob/main/blog/2019-08-09-the-downsides-of-json-for-config-files/index.md","source":"@site/blog/2019-08-09-the-downsides-of-json-for-config-files/index.md","title":"<\u8bd1>JSON\u683c\u5f0f\u4f5c\u4e3a\u914d\u7f6e\u6587\u4ef6\u7684\u7f3a\u70b9","description":"\u7ffb\u8bd1\u81ea\u8fd9\u7bc7\u6587\u7ae0","date":"2019-08-09T00:00:00.000Z","formattedDate":"2019\u5e748\u67089\u65e5","tags":[{"label":"json","permalink":"/blog/tags/json"}],"readingTime":7.66,"truncated":false,"authors":[{"name":"Junlin Zhou","title":"Fullstack Engineer @ ZJU ICI","url":"https://github.com/edwardzjl","imageURL":"https://github.com/edwardzjl.png","key":"jlzhou"}],"frontMatter":{"slug":"the-downsides-of-json-for-config-files","title":"<\u8bd1>JSON\u683c\u5f0f\u4f5c\u4e3a\u914d\u7f6e\u6587\u4ef6\u7684\u7f3a\u70b9","authors":["jlzhou"],"tags":["json"]},"nextItem":{"title":"\u7cfb\u7edf\u4e2d\u72b6\u6001\u4e3a static \u7684\u670d\u52a1","permalink":"/blog/static-service"}},"content":"\u7ffb\u8bd1\u81ea[\u8fd9\u7bc7\u6587\u7ae0][1]\\n\\n\u6211\u6700\u8fd1\u63a5\u89e6\u5230\u8bb8\u591a\u9879\u76ee\u5c06 `JSON` \u7528\u4f5c\u914d\u7f6e\u6587\u4ef6\u3002\u6211\u8ba4\u4e3a\u8fd9\u4e0d\u662f\u4e00\u4e2a\u597d\u4e3b\u610f\u3002\\n\\n`JSON` \u4ece\u8bbe\u8ba1\u4e4b\u521d\u5c31\u4e0d\u662f\u7528\u4e8e\u505a\u914d\u7f6e\u6587\u4ef6\u7684\uff0c\u8fd9\u4e5f\u4e0d\u662f\u5b83\u64c5\u957f\u7684\u9886\u57df\u3002`JSON` \u7684\u76ee\u6807\u662f \\"\u8f7b\u91cf\u7ea7\u6570\u636e\u4ea4\u6362\u683c\u5f0f\\", \u540c\u65f6\u5177\u6709 \\"\u6613\u4e8e\u4eba\u7c7b\u8bfb\u5199\\", \\"\u6613\u4e8e\u4ee3\u7801\u89e3\u6790\u548c\u751f\u6210\\" \u7684\u7279\u70b9\u3002\u5b83\u5728\u5bf9 \\"\u4eba\u7c7b\u800c\u8a00\u7684\u4fbf\u5229\u6027\\" \u548c \\"\u5bf9\u673a\u5668\u800c\u8a00\u7684\u4fbf\u5229\u6027\\" \u4e4b\u95f4\u53d6\u5f97\u4e86\u8f83\u597d\u7684\u5e73\u8861, \u5728\u8bb8\u591a\u5e94\u7528\u573a\u666f\u4e0b\u90fd\u662f\u6bd4 `XML` \u66f4\u597d\u7684\u66ff\u4ee3\u65b9\u6848\u3002\\n\\n\u7136\u800c\uff0c\u5c06 `JSON` \u7528\u4e8e\u5176\u4ed6\u76ee\u7684\u6709\u70b9\u7c7b\u4f3c\u4e8e\u8bf4 \\"\u563f\uff0c\u8fd9\u628a\u9524\u5b50\u975e\u5e38\u9002\u5408\u9489\u9489\u5b50\uff01\u6211\u559c\u6b22\u5b83\uff01\u4e3a\u4ec0\u4e48\u4e0d\u7528\u5b83\u6765\u62e7\u87ba\u4e1d\uff01\\" \u5f53\u7136\u5b83\u4e0d\u662f\u5b8c\u5168\u4e0d\u80fd\u7528\uff0c\u53ea\u662f\u4e0d\u5408\u9002\u505a\u8fd9\u6837\u7684\u5de5\u4f5c\u3002\\n\\n\u76ee\u524d\u4e3a\u6b62\uff0c\u5c06 `JSON` \u7528\u4f5c\u5176\u5b83\u7528\u9014\u6700\u5927\u7684\u95ee\u9898\u5728\u4e8e\u4e0d\u80fd\u5728 `JSON` \u6587\u4ef6\u4e2d\u6dfb\u52a0\u6ce8\u91ca\u3002\u67d0\u4e9b\u7279\u5b9a\u7684 `JSON` \u89e3\u6790\u5668\u652f\u6301\u5728 `JSON` \u4e2d\u6dfb\u52a0\u6ce8\u91ca\uff0c\u4f46\u662f\u7edd\u5927\u90e8\u5206\u7684\u89e3\u6790\u5668\u90fd\u4e0d\u652f\u6301\u3002`JSON` \u7684\u53d1\u660e\u8005 `Douglas Crockford` \u58f0\u79f0 `JSON` \u6700\u5f00\u59cb\u662f\u652f\u6301\u6ce8\u91ca\u7684\uff0c\u7136\u800c\u7531\u4e8e\u4e00\u4e9b\u539f\u56e0\uff0c\u4ed6\u7279\u610f\u79fb\u9664\u4e86\u5bf9\u6ce8\u91ca\u7684\u652f\u6301\u3002\u60f3\u8981\u6df1\u5165\u7814\u7a76\u7684\u670b\u53cb\u53ef\u4ee5\u770b[\u8fd9\u91cc][2]\u3002\\n\\n\u6211\u4eec\u5728\u5199\u914d\u7f6e\u6587\u4ef6\u65f6\u7ecf\u5e38\u4f1a\u9047\u5230\u9700\u8981\u6dfb\u52a0\u6ce8\u91ca\u7684\u573a\u666f\u3002\u4f8b\u5982\u89e3\u91ca\u4e3a\u4ec0\u4e48\u5c06\u914d\u7f6e\u9879\u8bbe\u7f6e\u4e3a\u5f53\u524d\u7684\u503c\uff0c\u6dfb\u52a0\u4e00\u4e9b\u52a9\u8bb0\u7b26\u6216\u662f\u6ce8\u610f\u4e8b\u9879\uff0c\u5bf9\u4e8e\u9519\u8bef\u914d\u7f6e\u7684\u8b66\u544a\uff0c\u5728\u6587\u4ef6\u4e2d\u4fdd\u5b58\u4e00\u4efd\u57fa\u7840\u7684 `changelog`\uff0c\u53c8\u6216\u5355\u7eaf\u662f\u5728debug\u65f6\u9700\u8981\u6ce8\u91ca\u6389\u4e00\u4e9b\u914d\u7f6e\u9879\u3002\\n\\n\u4e00\u4e2a\u53ef\u884c\u7684\u89e3\u51b3\u65b9\u6cd5\u662f\u5c06\u539f\u672c\u7684\u6570\u636e\u5b58\u50a8\u5728\u4e00\u4e2a object \u4e2d\uff0c\u5728\u8fd9\u4e2a object \u4e2d\u901a\u8fc7\u4e24\u4e2a\u6761\u76ee\u5206\u522b\u5b58\u50a8\u6570\u636e\u548c\u6ce8\u91ca\u3002\u4f8b\u5982\u539f\u672c\u7684\u914d\u7f6e\u6587\u4ef6\u5982\u4e0b\uff1a\\n\\n```json\\n{\\n  \\"config_name\\": \\"config_value\\"\\n}\\n```\\n\\n\u4fee\u6539\u540e\u53d8\u6210\u5982\u4e0b\u5f62\u5f0f:\\n\\n```json\\n{\\n  \\"config_name\\": {\\n\\t  \\"actual_data\\": \\"config_value\\",\\n\\t\\t\\"comment\\": \\"a comment\\"\\n  }\\n}\\n```\\n\\n\u4f46\u662f\u5728\u6211\u770b\u6765\u8fd9\u79cd\u65b9\u5f0f\u4e11\u7684\u538b\u6279\u3002\\n\\n\u8fd8\u6709\u4e00\u4e9b\u4eba\u6307\u51fa\u53ef\u4ee5\u901a\u8fc7 commit log \u7684\u5f62\u5f0f\u6765\u5b9e\u73b0\u6ce8\u91ca *\uff08\u8bd1\u8005\uff1a\u4e0d\u6e05\u695a\u4ed6\u8fd9\u91cc\u6307\u7684\u662f\u4e0d\u662f git commit log\uff0c\u5982\u679c\u662f\u7684\u8bdd\u628a\u8fd9\u4e2a\u5f53\u4f5c\u6ce8\u91ca\u65b9\u5f0f\u597d\u50cf\u5341\u5206\u96be\u7528\u5427\uff1f\uff09*\uff0c\u4f46\u662f\u53c8\u6709\u51e0\u4e2a\u4eba\u4f1a\u53bb\u7ec6\u8bfb commit history\uff1f\\n\\n\u4e00\u4e9b\u57fa\u4e8e `JSON` \u8fdb\u884c\u6269\u5c55\u7684\u683c\u5f0f\uff0c\u4f8b\u5982 `JSON5`\uff0c`Hjson` \u548c `HOCON`\uff0c\u4ee5\u53ca\u4e00\u5c0f\u90e8\u5206 `JSON` \u89e3\u6790\u5668\u6dfb\u52a0\u4e86\u5bf9\u6ce8\u91ca\u7684\u652f\u6301\u3002\u8fd9\u5f88\u5b9e\u7528\uff0c\u4f46\u8fd9\u4e9b\u90fd\u5c5e\u4e8e `JSON` \u7684\u53d8\u79cd\uff0c\u56e0\u6b64\u4e0d\u5728\u672c\u7bc7\u7684\u8ba8\u8bba\u8303\u56f4\u4e4b\u5185\u3002\\n\\n\u540c\u65f6\u6211\u4e5f\u53d1\u73b0\u624b\u5de5\u7f16\u8f91 `JSON` \u7684\u7528\u6237\u4f53\u9a8c\u4e0d\u662f\u90a3\u4e48\u53cb\u597d\uff1a\u4f60\u5f97\u7559\u610f\u884c\u5c3e\u662f\u5426\u8981\u6dfb\u52a0\u9017\u53f7\uff0c\u5f97\u4e86\u89e3\u7528\u4e0d\u7528\u5f15\u53f7\u5bf9\u542b\u4e49\u7684\u5f71\u54cd\uff0c\u540c\u65f6 `JSON` \u4e5f\u4e0d\u652f\u6301\u5b57\u7b26\u4e32\u5185\u6362\u884c\u3002\u8fd9\u4e9b\u7279\u6027\u5bf9\u4e8e \\"\u8f7b\u91cf\u7ea7\u6570\u636e\u4ea4\u6362\u683c\u5f0f\\" \u800c\u8a00\u4e0d\u662f\u574f\u4e8b\uff0c\u4f46\u662f\u5bf9\u4e8e\u7f16\u8f91\u914d\u7f6e\u6587\u4ef6\u8fd9\u4ef6\u4e8b\u6765\u8bf4\u5374\u4e0d\u662f\u90a3\u4e48\u53ef\u7231\u3002\u603b\u7684\u6765\u8bf4\uff0c\u5c06 `JSON` \u7528\u4f5c\u914d\u7f6e\u6587\u4ef6\u867d\u7136\u53ef\u884c\uff0c\u4f46\u5e76\u4e0d\u4f18\u96c5\u3002\\n\\nMediaWiki \u7684\u65b0\u6269\u5c55\u7cfb\u7edf\u4fc3\u4f7f\u6211\u5199\u4e0b\u8fd9\u7bc7\u6587\u7ae0\u3002\u65e7\u7684\u7cfb\u7edf\u901a\u8fc7 PHP \u6587\u4ef6\u6765\u6302\u63a5\u6838\u5fc3\u4ee3\u7801\uff0c\u52a0\u8f7d\u6240\u9700\u7684\u4f9d\u8d56\u9879\u7b49\u3002\u65b0\u7cfb\u7edf\u901a\u8fc7 JSON \u6587\u4ef6\u5b9e\u73b0\u8fd9\u4e9b\u914d\u7f6e\u3002\u8fd9\u6837\u7684\u66f4\u65b0\u635f\u5931\u4e86 PHP \u90a3\u79cd\u80fd\u591f\u5de7\u5999\u89e3\u51b3\u4e0e\u5176\u4ed6\u63d2\u4ef6\u517c\u5bb9\u6027\u7684\u80fd\u529b\u3002 *\uff08\u8fd9\u6bb5\u6ca1\u770b\u61c2\uff09*\\n\\n\u540c\u65f6\u5b83\u4e5f\u5e26\u6765\u4e86\u66f4\u591a\u5b9e\u73b0\u590d\u6742\u5ea6\u3002\u65e7\u7684\u7cfb\u7edf\u5728\u5f15\u5165\u914d\u7f6e\u6587\u4ef6\u65f6\u4ec5\u4ec5\u9700\u8981\u4e00\u884c\u4ee3\u7801\uff1a\\n\\n```javascript\\nrequire(\'plugin/foo/plugin.php\');\\n```\\n\\n\u800c\u65b0\u7cfb\u7edf\u5374\u9700\u8981\u5bf9 JSON \u6587\u4ef6\u7684\u5185\u5bb9\u8fdb\u884c\u89e3\u6790\u3002\u8fd9\u5728\u63d0\u5347\u5b9e\u73b0\u590d\u6742\u5ea6\u7684\u540c\u65f6\uff0c\u4e5f\u63d0\u9ad8\u4e86 debug \u7684\u96be\u5ea6\u3002\\n*\uff08\u8fd9\u6bb5\u4e0d\u592a\u8d5e\u540c\uff0cXML \u4f5c\u4e3a\u914d\u7f6e\u6587\u4ef6\uff0c\u540c\u6837\u8981\u8fdb\u884c\u89e3\u6790\uff0c\u8fd9\u4e0d\u662f JSON \u7684\u95ee\u9898\u3002\uff09*\\n\\n\u4f7f\u7528 JSON \u6587\u4ef6\u5b58\u50a8\u57fa\u672c\u5143\u6570\u636e\u662f\u53ef\u884c\u7684\uff08\u66f4\u5bb9\u6613\u89e3\u6790\u4ee5\u53ca\u5728\u7f51\u7ad9\u4e0a\u663e\u793a\uff09\uff0c\u4f46\u4f7f\u7528\u5b83\u6765\u63cf\u8ff0\u4ee3\u7801\u7684\u5de5\u4f5c\u65b9\u5f0f\u5bf9\u6211\u6765\u8bf4\u662f\u6ee5\u7528 DC\uff08Declarative configuration \uff0c\u58f0\u660e\u6027\u914d\u7f6e\uff09\u3002\u6bd5\u7adf\uff0c\u8fd9\u662f\u4ee3\u7801\u7684\u5de5\u4f5c\u3002\\n\\n\u8bb8\u591a\u4eba\u95ee\u6211\u90a3\u5230\u5e95\u8be5\u7528\u4ec0\u4e48(\u6765\u505a\u914d\u7f6e\u6587\u4ef6)\uff0c\u8fd9\u5176\u5b9e\u662f\u4e2a\u5f88\u590d\u6742\u7684\u95ee\u9898\uff0c\u5173\u7cfb\u5230\u4f60\u7a0b\u5e8f\u7684\u5e94\u7528\u573a\u666f\u3001\u7f16\u7a0b\u8bed\u8a00\u3001\u5e93\u73af\u5883\u751a\u81f3\u4e00\u4e9b\u793e\u4ea4\u56e0\u7d20\uff08\uff1f\uff09\u3002\u6700\u597d\u7684\u56de\u7b54\u53ef\u80fd\u5c31\u662f\u201c\u627e\u5230\u80fd\u7b26\u5408\u4f60\u9700\u6c42\u7684\u6700\u7b80\u5355\u7684\u65b9\u6848\u201d\u3002\\n\\n\u6709\u4e00\u4e9b JSON \u7684\u6269\u5c55\u683c\u5f0f\u66f4\u9002\u5408\u4e8e\u4eba\u7c7b\u8fdb\u884c\u7f16\u8f91\uff0c\u4f8b\u5982 JSON5\u3001Hjson \u548c HOCON\u3002\u8fd9\u4e9b\u770b\u8d77\u6765\u90fd\u662f\u666e\u901aJSON\u7684\u5408\u7406\u5347\u7ea7\uff0c\u5c3d\u7ba1\u6211\u81ea\u5df1\u6ca1\u6709\u4f7f\u7528\u8fc7\u5b83\u4eec\u3002\u7279\u522b\u662f JSON5 \u4f3c\u4e4e\u662f\u4e00\u4e2a\u4e0d\u9519\u7684\u9009\u62e9\uff0c\u56e0\u4e3a\u5b83\u5bf9 JSON \u7684\u6539\u52a8\u6700\u5c11\u3002\u6211\u4e0d\u80fd\u7ed9\u51fa\u5173\u4e8e\u8fd9\u4e9b\u6269\u5c55\u683c\u5f0f\u7684\u5efa\u8bae\uff0c\u56e0\u4e3a\u6211\u6ca1\u6709\u6240\u6709\u7684\u683c\u5f0f\u8fdb\u884c\u6df1\u5165\u7684\u6bd4\u8f83\u3002\u53ea\u662f\u770b\u4e00\u773c\u683c\u5f0f\u89c4\u8303\u5e76\u4e0d\u80fd\u53d1\u73b0\u6f5c\u5728\u7684\u7f3a\u70b9\uff08YAML \u5c31\u662f\u4e00\u4e2a\u5f88\u597d\u7684\u4f8b\u5b50\uff09\u3002\u6211\u6ca1\u6709\u65f6\u95f4\u6216\u662f\u5174\u8da3\u5bf9\u6240\u6709\u66ff\u4ee3\u65b9\u6848\u8fdb\u884c\u5168\u9762\u6df1\u5165\u7684\u5ba1\u67e5\u3002\\n\\n## \u540e\u8bb0\\n\\n\u8fd9\u662f\u6211\u7b2c\u4e00\u6b21\u505a\u9700\u8981\u53d1\u5e03\u5230\u7f51\u4e0a\u7684\u6bd4\u8f83\u6b63\u5f0f\u7684\u7ffb\u8bd1\u5de5\u4f5c\u3002\u867d\u7136\u6700\u65e9\u81ea\u5df1\u5728\u8bfb paper \u7684\u65f6\u5019\u56e0\u4e3a\u82f1\u8bed\u751f\u758f\uff0c\u4e5f\u4f1a\u8fb9\u8bfb\u8fb9\u7ffb\u8bd1\u4e00\u4e9b\uff0c\u4f46\u662f\u6bd5\u7adf\u90a3\u662f\u7ffb\u8bd1\u7ed9\u81ea\u5df1\u770b\u7684\uff0c\u53ea\u8981\u81ea\u5df1\u80fd\u770b\u61c2\u5c31\u884c\u4e86\uff0c\u4e5f\u4e0d\u7528\u8ffd\u6c42\u4ec0\u4e48\u8bed\u53e5\u901a\u987a\u4e4b\u7c7b\u7684\u3002\u7136\u800c\u8981\u53d1\u5e03\u51fa\u6765\u7684\u6587\u7ae0\u4e0d\u4e00\u6837\uff0c\u81f3\u5c11\u8981\u4fdd\u8bc1\u5927\u591a\u6570\u8bfb\u8005\u80fd\u591f\u770b\u5f97\u61c2\u3002\\n\\n\u6574\u7bc7\u7ffb\u5b8c\u56de\u8fc7\u5934\u770b\u770b\uff0c\u8fd8\u662f\u6709\u5f88\u591a\u751f\u786c\u4f3c\u673a\u7ffb\u7684\u5730\u65b9\uff0c\u4e3b\u8981\u539f\u56e0\u53ef\u80fd\u8fd8\u662f\u81ea\u5df1\u7684\u8868\u8fbe\u80fd\u529b\u4e0d\u591f\u3002\u7ffb\u8bd1\u6280\u672f\u6587\u7ae0\u5728\u6211\u770b\u6765\u662f\u4e2a\u5403\u529b\u4e0d\u8ba8\u597d\u7684\u6d3b\uff0c\u7ffb\u7684\u518d\u597d\u4e5f\u4e0d\u5982\u76f4\u63a5\u8bfb\u539f\u6587\u6765\u7684\u6e05\u6670\u3002\u81f3\u4e8e\u4e3a\u4ec0\u4e48\u8981\u505a\u8fd9\u6837\u7684\u4e8b\u60c5\uff0c \u6211\u60f3\u6709\u65f6\u95f4\u5355\u72ec\u5199\u4e00\u7bc7\u8c08\u4e00\u8c08\u3002\u76ee\u524d\u6765\u770b\uff0c\u5c31\u6743\u5f53\u662f\u5bf9\u4e8e\u81ea\u5df1\u8868\u8fbe\u80fd\u529b\u7684\u953b\u70bc\u5427\u3002\\n\\n[1]: https://www.arp242.net/json-config.html \\"The downsides of JSON for config files\\"\\n[2]: https://vorba.ch/2013/json-comments.html \\"Why are comments not allowed in JSON?\\""},{"id":"static-service","metadata":{"permalink":"/blog/static-service","editUrl":"https://github.com/edwardzjl/edwardzjl.github.io/blob/main/blog/2019-07-04-static-service/index.md","source":"@site/blog/2019-07-04-static-service/index.md","title":"\u7cfb\u7edf\u4e2d\u72b6\u6001\u4e3a static \u7684\u670d\u52a1","description":"\u6700\u8fd1\u5f00\u59cb\u63a5\u89e6 Linux \u8fd0\u7ef4\u7684\u5de5\u4f5c\uff0c\u7b2c\u4e00\u4ef6\u4e8b\u60c5\u5c31\u662f\u770b\u770b\u7cfb\u7edf\u4e2d\u8dd1\u4e86\u591a\u5c11\u670d\u52a1\u3002","date":"2019-07-04T00:00:00.000Z","formattedDate":"2019\u5e747\u67084\u65e5","tags":[{"label":"linux","permalink":"/blog/tags/linux"}],"readingTime":0.87,"truncated":false,"authors":[{"name":"Junlin Zhou","title":"Fullstack Engineer @ ZJU ICI","url":"https://github.com/edwardzjl","imageURL":"https://github.com/edwardzjl.png","key":"jlzhou"}],"frontMatter":{"slug":"static-service","title":"\u7cfb\u7edf\u4e2d\u72b6\u6001\u4e3a static \u7684\u670d\u52a1","authors":["jlzhou"],"tags":["linux"]},"prevItem":{"title":"<\u8bd1>JSON\u683c\u5f0f\u4f5c\u4e3a\u914d\u7f6e\u6587\u4ef6\u7684\u7f3a\u70b9","permalink":"/blog/the-downsides-of-json-for-config-files"},"nextItem":{"title":"<\u8bd1> javax.persistence.Id \u548c org.springframework.data.annotation.Id \u7684\u533a\u522b","permalink":"/blog/difference-between-javax.persistence.id-and-org.springframework.data.annotation.id"}},"content":"\u6700\u8fd1\u5f00\u59cb\u63a5\u89e6 Linux \u8fd0\u7ef4\u7684\u5de5\u4f5c\uff0c\u7b2c\u4e00\u4ef6\u4e8b\u60c5\u5c31\u662f\u770b\u770b\u7cfb\u7edf\u4e2d\u8dd1\u4e86\u591a\u5c11\u670d\u52a1\u3002\\n\\n\u96c6\u7fa4\u7528\u7684\u662f CentOS 7\uff0c\u53ef\u4ee5\u901a\u8fc7 ```bash systemctl list-unit-files``` \u8fd9\u4e2a\u547d\u4ee4\u67e5\u770b\u6240\u6709\u670d\u52a1\uff0c\u6572\u4e0b\u56de\u8f66\u540e\u6253\u5370\u51fa\u6765\u8fd9\u4e48\u4e00\u5806\u73a9\u5e94\u513f\uff1a\\n\\n![services](./services.png \\"services\\")\\n\\nservice \u7684 `disabled` \u548c `enabled` \u72b6\u6001\u90fd\u597d\u7406\u89e3\uff0c`static` \u662f\u4e2a\u5565\uff1f\u5728[\u4e0d\u5b58\u5728\u7684\u7f51\u7ad9][1]\u4e0a\u4e00\u987f\u67e5\u627e\uff0c\u627e\u5230\u5982\u4e0b\u8fd9\u756a\u89e3\u91ca\uff1a\\n\\n> \\"static\\" means \\"enabled because something else wants it\\". Think by analogy to pacman\'s package install reasons:\\n>\\n> - enabled :: explicitly installed\\n> - static :: installed as dependency\\n> - disabled :: not installed\\n\\n\u610f\u601d\u662f\uff0c\u72b6\u6001\u4e3a `static` \u7684\u670d\u52a1\uff0c\u662f\u4f5c\u4e3a\u522b\u7684\u670d\u52a1\u7684\u4f9d\u8d56\u800c\u5b58\u5728\u3002\\n\\n[1]: https://bbs.archlinux.org/viewtopic.php?id=147964 \\"systemd \'static\' unit file state\\""},{"id":"difference-between-javax.persistence.id-and-org.springframework.data.annotation.id","metadata":{"permalink":"/blog/difference-between-javax.persistence.id-and-org.springframework.data.annotation.id","editUrl":"https://github.com/edwardzjl/edwardzjl.github.io/blob/main/blog/2019-06-27-difference-between-javax.persistence.id-and-org.springframework.data.annotation.id/index.md","source":"@site/blog/2019-06-27-difference-between-javax.persistence.id-and-org.springframework.data.annotation.id/index.md","title":"<\u8bd1> javax.persistence.Id \u548c org.springframework.data.annotation.Id \u7684\u533a\u522b","description":"org.springframework.data.annotation.Id","date":"2019-06-27T00:00:00.000Z","formattedDate":"2019\u5e746\u670827\u65e5","tags":[{"label":"spring","permalink":"/blog/tags/spring"},{"label":"java","permalink":"/blog/tags/java"}],"readingTime":0.405,"truncated":false,"authors":[{"name":"Junlin Zhou","title":"Fullstack Engineer @ ZJU ICI","url":"https://github.com/edwardzjl","imageURL":"https://github.com/edwardzjl.png","key":"jlzhou"}],"frontMatter":{"slug":"difference-between-javax.persistence.id-and-org.springframework.data.annotation.id","title":"<\u8bd1> javax.persistence.Id \u548c org.springframework.data.annotation.Id \u7684\u533a\u522b","authors":["jlzhou"],"tags":["spring","java"]},"prevItem":{"title":"\u7cfb\u7edf\u4e2d\u72b6\u6001\u4e3a static \u7684\u670d\u52a1","permalink":"/blog/static-service"},"nextItem":{"title":"Install postgres on OSX","permalink":"/blog/install-postgres-on-osx"}},"content":"## org.springframework.data.annotation.Id\\n\\n`org.springframework.data.annotation.Id` \u662f Spring \u5b9a\u4e49\u7684 annotation\uff0c\u7528\u6765\u652f\u6301 \\"\u6ca1\u6709\u50cf JPA \u90a3\u6837\u7684\u6301\u4e45\u5316 API\\" \u7684\u975e\u5173\u7cfb\u578b\u6570\u636e\u5e93\u6216\u662f\u6846\u67b6\u7684\u6301\u4e45\u5316\uff0c\u56e0\u6b64\u5b83\u5e38\u88ab\u7528\u4e8e\u5176\u5b83 spring-data \u9879\u76ee\uff0c\u4f8b\u5982 spring-data-mongodb \u548c spring-data-solr \u7b49\u3002\\n\\n## javax.persistence.Id\\n\\n`javax.persistence.Id` \u662f\u7531 JPA \u5b9a\u4e49\u7684 annotation\uff0cJPA \u4ec5\u9002\u7528\u4e8e\u5173\u7cfb\u6570\u636e\u7684\u7ba1\u7406\u3002\\n\\n\\n## Ref\\n\\n- [whats-the-difference-between-javax-persistence-id-and-org-springframework-data](https://stackoverflow.com/questions/39643960/whats-the-difference-between-javax-persistence-id-and-org-springframework-data)"},{"id":"install-postgres-on-osx","metadata":{"permalink":"/blog/install-postgres-on-osx","editUrl":"https://github.com/edwardzjl/edwardzjl.github.io/blob/main/blog/2019-04-13-install-postgres-on-osx/index.md","source":"@site/blog/2019-04-13-install-postgres-on-osx/index.md","title":"Install postgres on OSX","description":"If you installed Postgres from homebrew, the default user postgres isn\'t automatically created, you need to run following command in your terminal:","date":"2019-04-13T00:00:00.000Z","formattedDate":"2019\u5e744\u670813\u65e5","tags":[{"label":"postgres","permalink":"/blog/tags/postgres"},{"label":"osx","permalink":"/blog/tags/osx"}],"readingTime":0.135,"truncated":false,"authors":[{"name":"Junlin Zhou","title":"Fullstack Engineer @ ZJU ICI","url":"https://github.com/edwardzjl","imageURL":"https://github.com/edwardzjl.png","key":"jlzhou"}],"frontMatter":{"slug":"install-postgres-on-osx","title":"Install postgres on OSX","authors":["jlzhou"],"tags":["postgres","osx"]},"prevItem":{"title":"<\u8bd1> javax.persistence.Id \u548c org.springframework.data.annotation.Id \u7684\u533a\u522b","permalink":"/blog/difference-between-javax.persistence.id-and-org.springframework.data.annotation.id"},"nextItem":{"title":"vim 8.0 \u526a\u5207\u677f\u8bbe\u7f6e","permalink":"/blog/config-vim-8-clipboard"}},"content":"If you installed Postgres from homebrew, the default user `postgres` isn\'t automatically created, you need to run following command in your terminal:\\n\\n```sh\\n/Applications/Postgres.app/Contents/Versions/9.*/bin/createuser -s postgres\\n```"},{"id":"config-vim-8-clipboard","metadata":{"permalink":"/blog/config-vim-8-clipboard","editUrl":"https://github.com/edwardzjl/edwardzjl.github.io/blob/main/blog/2019-03-14-config-vim-8-clipboard/index.md","source":"@site/blog/2019-03-14-config-vim-8-clipboard/index.md","title":"vim 8.0 \u526a\u5207\u677f\u8bbe\u7f6e","description":"\u901a\u8fc7 ubuntu \u548c centos \u7684\u6e90\u5b89\u88c5\u7684 vim \u7248\u672c\u8f83\u8001\uff08\u597d\u50cf\u662f7.4.x\uff09","date":"2019-03-14T00:00:00.000Z","formattedDate":"2019\u5e743\u670814\u65e5","tags":[{"label":"vim","permalink":"/blog/tags/vim"}],"readingTime":0.71,"truncated":false,"authors":[{"name":"Junlin Zhou","title":"Fullstack Engineer @ ZJU ICI","url":"https://github.com/edwardzjl","imageURL":"https://github.com/edwardzjl.png","key":"jlzhou"}],"frontMatter":{"slug":"config-vim-8-clipboard","title":"vim 8.0 \u526a\u5207\u677f\u8bbe\u7f6e","authors":["jlzhou"],"tags":["vim"]},"prevItem":{"title":"Install postgres on OSX","permalink":"/blog/install-postgres-on-osx"}},"content":"\u901a\u8fc7 `ubuntu` \u548c `centos` \u7684\u6e90\u5b89\u88c5\u7684 `vim` \u7248\u672c\u8f83\u8001\uff08\u597d\u50cf\u662f7.4.x\uff09\\n\\n8.0 \u4e4b\u540e\u7684 `vim`\uff0c\u5b98\u7f51\u63a8\u8350\u7684\u5b89\u88c5\u65b9\u5f0f\u662f\u4ece git clone \u6e90\u7801\u7f16\u8bd1\\n\\n\u9ed8\u8ba4\u7f16\u8bd1\u51fa\u6765\u7684 `vim` \u662f\u6ca1\u6709 clipboard support \u7684\uff0c\u65e0\u6cd5\u901a\u8fc7\u5bc4\u5b58\u5668\u4e0e\u7cfb\u7edf\u526a\u5207\u677f\u8fdb\u884c\u4ea4\u4e92\\n\\n\u5728\u7f16\u8bd1\u65f6\u589e\u52a0 clip board support \u9700\u8981\u7684\u6700\u5c0f\u4f9d\u8d56\u4e3a `xorg header files` \u548c `x11 dbus`\\n\\n\u5728 <https://packages.ubuntu.com> \u91cc\u4e00\u901a\u641c\u7d22\u53d1\u73b0 `xorg header files` \u662f\u5728 `libx11-dev` \u8fd9\u4e2a\u5305\u91cc\uff0c\u800c `x11 dbus` \u5728 `dbus-x11`\\n\\n\u56e0\u6b64\u6574\u4e2a\u7f16\u8bd1\u8fc7\u7a0b\u5982\u4e0b\uff1a\\n\\n```sh\\nsudo apt-get install libx11-dev dbus-x11\\n./configure --with-features=huge\\nmake\\nsudo make install\\n```"}]}')}}]);