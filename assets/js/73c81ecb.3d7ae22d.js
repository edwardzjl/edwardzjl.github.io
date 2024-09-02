"use strict";(self.webpackChunkedwardzjl_github_io=self.webpackChunkedwardzjl_github_io||[]).push([[3005],{5680:(e,t,n)=>{n.d(t,{xA:()=>c,yg:()=>d});var r=n(6540);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var p=r.createContext({}),u=function(e){var t=r.useContext(p),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},c=function(e){var t=u(e.components);return r.createElement(p.Provider,{value:t},e.children)},s="mdxType",g={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,p=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),s=u(n),m=a,d=s["".concat(p,".").concat(m)]||s[m]||g[m]||o;return n?r.createElement(d,i(i({ref:t},c),{},{components:n})):r.createElement(d,i({ref:t},c))}));function d(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,i=new Array(o);i[0]=m;var l={};for(var p in t)hasOwnProperty.call(t,p)&&(l[p]=t[p]);l.originalType=e,l[s]="string"==typeof e?e:a,i[1]=l;for(var u=2;u<o;u++)i[u]=n[u];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},1470:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>p,contentTitle:()=>i,default:()=>g,frontMatter:()=>o,metadata:()=>l,toc:()=>u});var r=n(8168),a=(n(6540),n(5680));const o={slug:"config-vim-8-clipboard",title:"vim 8.0 \u526a\u5207\u677f\u8bbe\u7f6e",authors:["jlzhou"],tags:["vim"]},i=void 0,l={permalink:"/blog/config-vim-8-clipboard",editUrl:"https://github.com/edwardzjl/edwardzjl.github.io/blob/main/blog/2019-03-14-config-vim-8-clipboard/index.md",source:"@site/blog/2019-03-14-config-vim-8-clipboard/index.md",title:"vim 8.0 \u526a\u5207\u677f\u8bbe\u7f6e",description:"\u901a\u8fc7 ubuntu \u548c centos \u7684\u6e90\u5b89\u88c5\u7684 vim \u7248\u672c\u8f83\u8001\uff08\u597d\u50cf\u662f7.4.x\uff09",date:"2019-03-14T00:00:00.000Z",formattedDate:"2019\u5e743\u670814\u65e5",tags:[{label:"vim",permalink:"/blog/tags/vim"}],readingTime:.71,hasTruncateMarker:!1,authors:[{name:"Junlin Zhou",title:"Fullstack Engineer @ ZJU ICI",url:"https://github.com/edwardzjl",imageURL:"https://github.com/edwardzjl.png",key:"jlzhou"}],frontMatter:{slug:"config-vim-8-clipboard",title:"vim 8.0 \u526a\u5207\u677f\u8bbe\u7f6e",authors:["jlzhou"],tags:["vim"]},prevItem:{title:"Install postgres on OSX",permalink:"/blog/install-postgres-on-osx"}},p={authorsImageUrls:[void 0]},u=[],c={toc:u},s="wrapper";function g(e){let{components:t,...n}=e;return(0,a.yg)(s,(0,r.A)({},c,n,{components:t,mdxType:"MDXLayout"}),(0,a.yg)("p",null,"\u901a\u8fc7 ",(0,a.yg)("inlineCode",{parentName:"p"},"ubuntu")," \u548c ",(0,a.yg)("inlineCode",{parentName:"p"},"centos")," \u7684\u6e90\u5b89\u88c5\u7684 ",(0,a.yg)("inlineCode",{parentName:"p"},"vim")," \u7248\u672c\u8f83\u8001\uff08\u597d\u50cf\u662f7.4.x\uff09"),(0,a.yg)("p",null,"8.0 \u4e4b\u540e\u7684 ",(0,a.yg)("inlineCode",{parentName:"p"},"vim"),"\uff0c\u5b98\u7f51\u63a8\u8350\u7684\u5b89\u88c5\u65b9\u5f0f\u662f\u4ece git clone \u6e90\u7801\u7f16\u8bd1"),(0,a.yg)("p",null,"\u9ed8\u8ba4\u7f16\u8bd1\u51fa\u6765\u7684 ",(0,a.yg)("inlineCode",{parentName:"p"},"vim")," \u662f\u6ca1\u6709 clipboard support \u7684\uff0c\u65e0\u6cd5\u901a\u8fc7\u5bc4\u5b58\u5668\u4e0e\u7cfb\u7edf\u526a\u5207\u677f\u8fdb\u884c\u4ea4\u4e92"),(0,a.yg)("p",null,"\u5728\u7f16\u8bd1\u65f6\u589e\u52a0 clip board support \u9700\u8981\u7684\u6700\u5c0f\u4f9d\u8d56\u4e3a ",(0,a.yg)("inlineCode",{parentName:"p"},"xorg header files")," \u548c ",(0,a.yg)("inlineCode",{parentName:"p"},"x11 dbus")),(0,a.yg)("p",null,"\u5728 ",(0,a.yg)("a",{parentName:"p",href:"https://packages.ubuntu.com"},"https://packages.ubuntu.com")," \u91cc\u4e00\u901a\u641c\u7d22\u53d1\u73b0 ",(0,a.yg)("inlineCode",{parentName:"p"},"xorg header files")," \u662f\u5728 ",(0,a.yg)("inlineCode",{parentName:"p"},"libx11-dev")," \u8fd9\u4e2a\u5305\u91cc\uff0c\u800c ",(0,a.yg)("inlineCode",{parentName:"p"},"x11 dbus")," \u5728 ",(0,a.yg)("inlineCode",{parentName:"p"},"dbus-x11")),(0,a.yg)("p",null,"\u56e0\u6b64\u6574\u4e2a\u7f16\u8bd1\u8fc7\u7a0b\u5982\u4e0b\uff1a"),(0,a.yg)("pre",null,(0,a.yg)("code",{parentName:"pre",className:"language-sh"},"sudo apt-get install libx11-dev dbus-x11\n./configure --with-features=huge\nmake\nsudo make install\n")))}g.isMDXComponent=!0}}]);