/**
 * 对 AntDesign 主题进行改写，会覆盖 libs/ui 中的主题
 * https://ant.design/docs/react/customize-theme-cn
 * */
// import { theme } from 'antd';
import { AliasToken } from 'antd/es/theme/interface';
import { ComponentTokenMap } from 'antd/es/theme/interface/components';

// const { getDesignToken } = theme;
// const globalToken = getDesignToken();

const themeToken: Partial<AliasToken> = {
  colorPrimary: '#faad14',
};

const menuTheme: Partial<ComponentTokenMap['Menu']> = {
  darkItemHoverBg: themeToken.colorPrimary,
};

export const appThemeConfig = {
  themeToken,
  menuTheme,
};
