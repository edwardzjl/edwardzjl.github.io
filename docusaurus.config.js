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
        { to: '/', label: 'Blog', position: 'left' },
        {
          href: 'https://github.com/edwardzjl/edwardzjl.github.io',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: '/',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/edwardzjl/edwardzjl.github.io',
            },
          ],
        },
        {
          title: 'Feed',
          items: [
            {
              label: 'RSS',
              to: 'https://edwardzjl.github.io/rss.xml',
            },
            {
              label: 'Atom',
              href: 'https://edwardzjl.github.io/atom.xml',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} My Project, Inc. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  },
};

export default config;
