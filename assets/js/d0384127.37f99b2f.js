"use strict";(self.webpackChunkedwardzjl_github_io=self.webpackChunkedwardzjl_github_io||[]).push([[4024],{3905:function(e,t,r){r.d(t,{Zo:function(){return c},kt:function(){return f}});var n=r(7294);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function l(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},a=Object.keys(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var s=n.createContext({}),u=function(e){var t=n.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},c=function(e){var t=u(e.components);return n.createElement(s.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},g=n.forwardRef((function(e,t){var r=e.components,o=e.mdxType,a=e.originalType,s=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),g=u(r),f=o,m=g["".concat(s,".").concat(f)]||g[f]||p[f]||a;return r?n.createElement(m,i(i({ref:t},c),{},{components:r})):n.createElement(m,i({ref:t},c))}));function f(e,t){var r=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=r.length,i=new Array(a);i[0]=g;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:o,i[1]=l;for(var u=2;u<a;u++)i[u]=r[u];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}g.displayName="MDXCreateElement"},5347:function(e,t,r){r.r(t),r.d(t,{assets:function(){return c},contentTitle:function(){return s},default:function(){return f},frontMatter:function(){return l},metadata:function(){return u},toc:function(){return p}});var n=r(7462),o=r(3366),a=(r(7294),r(3905)),i=["components"],l={slug:"install-postgres-on-osx",title:"Install postgres on OSX",authors:["jlzhou"],tags:["postgres","osx"]},s=void 0,u={permalink:"/blog/install-postgres-on-osx",editUrl:"https://github.com/edwardzjl/edwardzjl.github.io/blob/main/blog/2019-04-13-install-postgres-on-osx/index.md",source:"@site/blog/2019-04-13-install-postgres-on-osx/index.md",title:"Install postgres on OSX",description:"If you installed Postgres from homebrew, the default user postgres isn't automatically created, you need to run following command in your terminal:",date:"2019-04-13T00:00:00.000Z",formattedDate:"2019\u5e744\u670813\u65e5",tags:[{label:"postgres",permalink:"/blog/tags/postgres"},{label:"osx",permalink:"/blog/tags/osx"}],readingTime:.135,truncated:!1,authors:[{name:"Junlin Zhou",title:"Fullstack Engineer @ ZJU ICI",url:"https://github.com/edwardzjl",imageURL:"https://github.com/edwardzjl.png",key:"jlzhou"}],frontMatter:{slug:"install-postgres-on-osx",title:"Install postgres on OSX",authors:["jlzhou"],tags:["postgres","osx"]},prevItem:{title:"<\u8bd1> javax.persistence.Id \u548c org.springframework.data.annotation.Id \u7684\u533a\u522b",permalink:"/blog/difference-between-javax.persistence.id-and-org.springframework.data.annotation.id"},nextItem:{title:"vim 8.0 \u526a\u5207\u677f\u8bbe\u7f6e",permalink:"/blog/config-vim-8-clipboard"}},c={authorsImageUrls:[void 0]},p=[],g={toc:p};function f(e){var t=e.components,r=(0,o.Z)(e,i);return(0,a.kt)("wrapper",(0,n.Z)({},g,r,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("p",null,"If you installed Postgres from homebrew, the default user ",(0,a.kt)("inlineCode",{parentName:"p"},"postgres")," isn't automatically created, you need to run following command in your terminal:"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-sh"},"/Applications/Postgres.app/Contents/Versions/9.*/bin/createuser -s postgres\n")))}f.isMDXComponent=!0}}]);