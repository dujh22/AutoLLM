import { lazy } from 'react';
import { createBrowserRouter } from 'react-router-dom';
import { IRouteItem } from '@/types';
import { ROUTE_URLS } from '@/config/route-urls.ts';

const MainLayout = lazy(() => import('@/layout'));
const HomePage = lazy(() => import('@/pages/home'));
const ChatPage = lazy(() => import('@/pages/chat'));
const EmptyPage = lazy(() => import('@/pages/empty'));

const routes: IRouteItem[] = [
  {
    path: ROUTE_URLS.home,
    element: <MainLayout />,
    children: [
      {
        path: ROUTE_URLS.home,
        element: <HomePage />,
      },
      {
        path: ROUTE_URLS.chat,
        element: <ChatPage />,
      },
      {
        path: '*',
        element: <EmptyPage />,
      },
    ],
  },
];

const router = createBrowserRouter(routes);

export default router;
