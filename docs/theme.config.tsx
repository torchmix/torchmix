
import React from 'react'
import { useRouter } from 'next/router'
import { useConfig } from 'nextra-theme-docs'
import { DocsThemeConfig } from 'nextra-theme-docs'

const config: DocsThemeConfig = {
  logo: <span><strong>TORCHMIX</strong></span>,
  project: {
    link: 'https://github.com/torchmix/torchmix',
  },
  // chat: {
  //   link: 'https://discord.com',
  // },
  docsRepositoryBase: 'https://github.com/torchmix/torchmix/blob/core/docs/',
  footer: {
    text: '',
  },
  i18n: [{locale:"ko",text:"kr"}],
  useNextSeoProps: () => {
    const { route } = useRouter()
    if (route !== "/"){
      return {
        titleTemplate:  '%s - torchmix'
      }
    }
    return {}
  },
  head: () => {
    const { asPath } = useRouter();
    const { frontMatter }= useConfig();
    return <>
      <meta property="og:url" content={`https://torchmix.github.io${asPath}`} />
      <meta property="og:title" content={frontMatter.title || 'torchmix'} />
      <meta property="og:description" content={frontMatter.description || 'The pytorch component library'} />
    </>
  },
  primaryHue: 196
}

export default config
