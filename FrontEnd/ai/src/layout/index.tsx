import { useState, useEffect, Suspense } from 'react';
import zhCN from 'antd/locale/zh_CN';
import { Outlet, useLocation } from 'react-router-dom';
import { ConfigProvider, theme, Layout, Menu, Button } from 'antd';
import { appThemeConfig } from '@/styles/themes.ts';
import { menus } from '@/config/menus.tsx';
import router from '@/config/routes';
import { MenuFoldOutlined, MenuUnfoldOutlined } from '@ant-design/icons';
import styles from './index.module.scss';
import SetupModal from '@/layout/components/setup-modal';

const MainLayout = () => {
  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken();
  const location = useLocation();

  const [modalType, setModalType] = useState<'edit' | 'new'>('new');
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedKey, setSelectedKey] = useState<string>(location.pathname);
  const [collapsed, setCollapsed] = useState(false);

  const onMenuSelect = (e: { selectedKeys: string[] }) => {
    const res = e.selectedKeys[0];
    router.navigate(res);
    setSelectedKey(res);
  };

  useEffect(() => {
    setSelectedKey(location.pathname);
  }, [location.pathname]);

  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          ...appThemeConfig.themeToken,
        },
      }}
    >
      <Layout style={{ minHeight: '100vh' }}>
        <Layout.Sider width={220} trigger={null} collapsible collapsed={collapsed} collapsedWidth={54}>
          <div className={styles.logo}>{import.meta.env.VITE_APP_APP_NAME}</div>
          <Menu className={styles.menu} theme="dark" selectedKeys={[selectedKey]} items={menus} onSelect={onMenuSelect} />
        </Layout.Sider>
        <Layout>
          <Layout.Header style={{ padding: 0, background: colorBgContainer }} className={styles.header}>
            <div className={styles.headerLeft}>
              <Button
                type="text"
                icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                onClick={() => setCollapsed(!collapsed)}
                style={{
                  fontSize: '16px',
                  width: 64,
                  height: 64,
                }}
              />
              <div className={styles.title}>{import.meta.env.VITE_APP_SLOGAN}</div>
            </div>
            <div className={styles.headerRight}>
              <Button
                type="text"
                onClick={() => {
                  setModalType('edit');
                  setModalVisible(true);
                }}
              >
                设置
              </Button>
            </div>
          </Layout.Header>
          <Layout.Content
            style={{
              margin: '24px 16px',
              padding: 24,
              minHeight: 280,
              background: colorBgContainer,
              borderRadius: borderRadiusLG,
            }}
          >
            <Suspense>
              <Outlet />
            </Suspense>
          </Layout.Content>
        </Layout>
      </Layout>
      <SetupModal visible={modalVisible} setVisible={setModalVisible} type={modalType} />
    </ConfigProvider>
  );
};

export default MainLayout;
