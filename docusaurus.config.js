import { themes as prismThemes } from 'prism-react-renderer';


const config = {
  title: 'Edwardzjl',
  tagline: '温故而知新',
  favicon: 'img/favicon.ico',

  url: 'https://edwardzjl.github.io',
  baseUrl: '/',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  organizationName: 'edwardzjl',
  projectName: 'edwardzjl.github.io',
  deploymentBranch: 'gh-pages',

  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['en', 'zh-Hans'],
  },

  markdown: {
    mdx1Compat: {
      comments: true,
      admonitions: false,
      headingIds: true,
    },
  },

  presets: [
    [
      'classic',
      {
        docs: false,
        blog: {
          routeBasePath: '/', // Serve the blog at the site's root
          showReadingTime: true,
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/edwardzjl/edwardzjl.github.io/blob/main/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'My Site',
      logo: {
        alt: 'My Site Logo',
        src: 'img/logo.svg',
      },
      items: [
        { label: 'Home', position: 'left', to: '/' },
        { label: 'Tags', position: 'left', to: '/tags' },
        {
          label: 'GitHub',
          position: 'right',
          href: 'https://github.com/edwardzjl/edwardzjl.github.io',
        },
        {
          label: 'Feed',
          position: 'right',
          type: 'dropdown',
          items: [
            {
              label: 'RSS',
              to: 'https://edwardzjl.github.io/rss.xml',
            },
            {
              label: 'Atom',
              to: 'https://edwardzjl.github.io/atom.xml',
            },
          ],
        },
      ],
    },
    footer: {
      style: 'dark',
      // links: [
      //   {
      //     title: 'More',
      //     items: [
      //       { label: 'Blog', to: '/' },
      //     ],
      //   },
      // ],
      copyright: `Copyright © ${new Date().getFullYear()} Junlin Zhou. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  },
};

export default config;
