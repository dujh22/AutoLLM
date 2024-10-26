import { CommentOutlined, HomeOutlined, BookOutlined, WechatOutlined } from '@ant-design/icons';
import { ROUTE_URLS } from './route-urls';
import { IMenuItem } from '@/types';

export const menus: IMenuItem[] = [
  {
    label: '首页',
    key: ROUTE_URLS.home,
    icon: <HomeOutlined />,
  },
  {
    label: '知识对话',
    key: ROUTE_URLS.chat,
    icon: <WechatOutlined />,
  },
  {
    label: '知识库',
    key: ROUTE_URLS.knowledge,
    icon: <BookOutlined />,
  },
  {
    label: '社区',
    key: ROUTE_URLS.community,
    icon: <CommentOutlined />,
  },
];
