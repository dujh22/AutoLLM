import { App } from 'antd';
import { RouterProvider } from 'react-router-dom';
import router from '@/config/routes.tsx';

function MyApp() {
  return (
    <App style={{ minHeight: '100vh' }}>
      <RouterProvider router={router} />
    </App>
  );
}

export default MyApp;
