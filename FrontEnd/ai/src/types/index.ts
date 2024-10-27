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
  videoPath?: string;
  linkPath?: string;
}


export interface IBriefResItem {
  content: string;
  file_path: string;
  generated_time: string;
  links: string[];
}

export interface IBriefRes {
  briefings: IBriefResItem | Array<IBriefResItem>
}

