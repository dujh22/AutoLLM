import React from 'react';

export interface IRouteItem {
  path: string;
  element: React.ReactElement;
  errorElement?: React.ReactElement;
  children?: IRouteItem[];
}

export interface IMenuItem {
  key: string;
  label: string;
  icon: React.ReactElement;
}

export interface IChatItem {
  id: string;
  content: string;
  type: 'question' | 'answer';
}
